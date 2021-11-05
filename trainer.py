import os
import time
import pickle as pkl
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.distributions.multivariate_normal import MultivariateNormal

from helpers.graph_helper import get_probs, sample_dist, convert_grid
from helpers.static_helper import bin_pred, f1_score


class Trainer:
    def __init__(self, num_epochs, early_stop_tolerance, clip, optimizer, loss_function,
                 learning_rate, weight_decay, momentum, device, save_dir, node2cell=None, regions=None, nodes=None,
                 plot_lr=True, coord_range=None, spatial_res=None):
        self.num_epochs = num_epochs
        self.clip = clip
        self.optimizer = optimizer
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.tolerance = early_stop_tolerance
        self.device = torch.device(device)
        self.loss_function = loss_function
        self.save_dir = save_dir
        self.plot_lr = plot_lr
        self.custom_losses = ["prob_loss", "likelihood"]
        self.regions = regions
        self.nodes = torch.from_numpy(nodes).to(self.device).float()
        self.coord_range = coord_range
        self.spatial_res = spatial_res

        if node2cell is not None:
            self.node2cell = {}
            for i, arr in node2cell.items():
                self.node2cell[i] = torch.from_numpy(arr).float().to(device)

        self.criterion_dict = {
            "MSE": nn.MSELoss(),
            "BCE": nn.BCELoss(),
            "likelihood": self.__likelihood_loss,
            "prob_loss": self.__prob_loss
        }

        self.model_step_preds = {key: [] for key in ["train", "val", "train_val", "test"]}
        self.model_step_labels = deepcopy(self.model_step_preds)

    def fit(self, model, batch_generator):
        model = model.to(self.device)
        model.train()

        if self.optimizer == "adam":
            optimizer = optim.Adam(model.parameters(),
                                   lr=self.learning_rate)
        else:
            optimizer = optim.SGD(model.parameters(),
                                  lr=self.learning_rate,
                                  momentum=self.momentum)

        train_loss = []
        val_loss = []
        tolerance = 0
        best_epoch = 0
        best_val_loss = 1e9
        best_dict = model.state_dict()
        for epoch in range(self.num_epochs):
            # train and validation loop
            start_time = time.time()

            # train
            running_train_loss = self.__step_loop(model=model,
                                                  generator=batch_generator,
                                                  mode='train',
                                                  optimizer=optimizer)

            # validation
            running_val_loss = self.__step_loop(model=model,
                                                generator=batch_generator,
                                                mode='val',
                                                optimizer=None)

            epoch_time = time.time() - start_time

            message_str = "Epoch: {}, Train_loss: {:.5f}, Validation_loss: {:.5f}, Took {:.3f} seconds."
            print(message_str.format(epoch + 1, running_train_loss, running_val_loss, epoch_time))

            # checkpoint
            self.__save_model(model)
            train_loss.append(running_train_loss)
            val_loss.append(running_val_loss)

            if running_val_loss < best_val_loss:
                best_epoch = epoch + 1
                best_val_loss = running_val_loss
                best_dict = deepcopy(model.state_dict())
                tolerance = 0
            else:
                tolerance += 1

            # perform predictions with the best model
            if tolerance > self.tolerance or epoch == self.num_epochs - 1:
                model.load_state_dict(best_dict)
                tr_loss, vl_loss, eval_loss = [self.__step_loop(model=model,
                                                                generator=batch_generator,
                                                                mode=mode,
                                                                optimizer=None,
                                                                collect_outputs=True) for mode in
                                               ["train", "val", "train_val"]]

                print("-*-" * 10)
                message_str = "Early exiting from epoch: {}, \nTrain Loss: {:5f}, Val Loss: {:5f}, Eval Loss: {:.5f}."
                print(message_str.format(best_epoch, tr_loss, vl_loss, eval_loss))
                print("-*-" * 10)

                # checkpoint
                self.__save_model(model)
                self.__save_outputs()
                break

            torch.cuda.empty_cache()

        if self.plot_lr:
            fig, ax = plt.subplots()
            ax.plot(list(range(epoch + 1)), train_loss, label="train_loss")
            ax.plot(list(range(epoch + 1)), val_loss, label="val_loss")
            ax.legend()
            ax.grid(True)
            ax.set_title(f"Learning Curve lr: {self.learning_rate:.5f}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_yscale("log")
            plt.show()

    def transform(self, model, batch_generator):
        test_loss = self.__step_loop(model=model,
                                     generator=batch_generator,
                                     mode='test',
                                     optimizer=None,
                                     collect_outputs=True)
        print('Test finished, test loss: {:.5f}'.format(test_loss))
        self.__save_model(model)
        self.__save_outputs()

    def __step_loop(self, model, generator, mode, optimizer, collect_outputs=False):
        running_loss = 0
        for idx, batch in enumerate(generator.generate(mode)):
            print('\r{}:{}/{}'.format(mode, idx, generator.num_iter(mode)), flush=True, end='')
            if idx + 1 == generator.num_iter(mode):
                calculate_score = True
            else:
                calculate_score = False
            loss = self.__step(model=model,
                               inputs=batch,
                               mode=mode,
                               optimizer=optimizer,
                               dataset_name=generator.dataset_name,
                               collect_outputs=collect_outputs,
                               calculate_score=calculate_score)
            running_loss += loss
        running_loss /= (idx + 1)
        return running_loss

    def __step(self, model, inputs, optimizer, mode, dataset_name, collect_outputs, calculate_score):
        if optimizer:
            optimizer.zero_grad()

        x, y = self.__prep_input(inputs[0]), self.__prep_input(inputs[1])
        if dataset_name == "graph":
            edge_index = inputs[2].to(self.device)
            pred = model.forward(x, edge_index)
        else:
            pred = model.forward(x)

        loss, pred = self.__get_loss(pred, y, calculate_score=calculate_score)

        if optimizer is not None:
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(model.parameters(), self.clip)

            # take step in classifier's optimizer
            optimizer.step()

        loss = loss.detach().cpu().numpy()
        # pred = pred.detach().cpu().numpy()
        # y = y.detach().cpu().numpy()
        #
        # if self.loss_function == "MSE":
        #     y = (y > 0).astype(int)
        #
        if collect_outputs:
            self.model_step_preds[mode].append(pred)
            self.model_step_labels[mode].append(y)

        # calc f1 score
        # pred = bin_pred(pred=pred.flatten(), label=y.flatten())
        # f1 = f1_score(y_true=y.flatten(), y_pred=pred)

        # print(f"Loss: {loss}, F1 Score: {f1}")
        print(f"Loss: {loss}")
        return loss

    def __get_loss(self, pred, y, **kwargs):
        if self.loss_function in self.custom_losses:
            loss, pred = self.criterion_dict[self.loss_function](pred=pred, y=y, **kwargs)
        else:
            loss = self.criterion_dict[self.loss_function](pred, y)

        return loss, pred

    def __prob_loss(self, pred, y):
        criterion = self.criterion_dict["BCE"]
        batch_prob = get_probs(pred, node2cell=self.node2cell)
        loss = criterion(batch_prob, y)
        return loss, batch_prob

    def __likelihood_loss(self, pred, y, calculate_score=False):
        pred_mu, pred_sigma = pred
        plt.figure()
        plt.scatter(pred_mu.detach().cpu().numpy()[0, :, 0], pred_mu.detach().cpu().numpy()[0, :, 1])
        plt.scatter(y[0].detach().cpu().numpy()[:, 0], y[0].detach().cpu().numpy()[:, 1])
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.show()
        total_loss = torch.tensor(0).to("cuda").float()
        counter = 0
        batch_dists = []
        for i in range(pred_mu.shape[0]):
            dists = []
            for j in range(pred_mu.shape[1]):
                mu = pred_mu[i, j]
                sigma = torch.eye(2).to("cuda") * pred_sigma[i, j]
                m = MultivariateNormal(mu.T, sigma)
                dists.append(m)
            for k in range(len(y[i])):
                nodes = y[i][k, 2:4]
                for n in nodes:
                    log_likes = dists[int(n)].log_prob(y[i][k, :2])
                    total_loss += -torch.sum(log_likes)
                counter += 1
            batch_dists.append(dists)
        total_loss /= counter

        dist_nodes = torch.sqrt(torch.sum((pred_mu - self.nodes) ** 2))
        total_loss += 0.1 * dist_nodes

        if calculate_score:
            grid_pred = sample_dist(batch_dists, coord_range=self.coord_range, grid_shape=self.spatial_res)
            grid_label = []
            for i in range(len(y)):
                grid = convert_grid(y[i][:, :2].detach().cpu().numpy(),
                                    coord_range=self.coord_range, grid_shape=self.spatial_res)
                grid = (grid > 0).astype(int)
                grid_label.append(grid)
            grid_label = np.stack(grid_label).flatten()
            grid_pred = bin_pred(grid_pred.flatten(), grid_label)
            score = f1_score(grid_label, grid_pred)
            print(f"F1 Score: {score:.5f}")

        return total_loss, pred

    def __prep_input(self, x):
        if isinstance(x, list):
            x = [x[i].float().to(self.device) for i in range(len(x))]
        else:
            x = x.float().to(self.device)
        return x

    def __save_model(self, model):
        model_path = os.path.join(self.save_dir, "model.pkl")
        with open(model_path, "wb") as f:
            pkl.dump(model, f)

    def __save_outputs(self):
        preds_path = os.path.join(self.save_dir, "preds.pkl")
        labels_path = os.path.join(self.save_dir, "labels.pkl")
        for path, obj in zip([preds_path, labels_path], [self.model_step_preds, self.model_step_labels]):
            with open(path, "wb") as f:
                pkl.dump(obj, f)
