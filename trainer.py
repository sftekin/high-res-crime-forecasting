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

from helpers.graph_helper import get_log_like, get_graph_stats, inverse_label
from helpers.static_helper import bin_pred, f1_score, confusion_matrix, accuracy_score


class Trainer:
    def __init__(self, num_epochs, early_stop_tolerance, clip, optimizer, loss_function,
                 learning_rate, weight_decay, momentum, device, save_dir, node2cell=None, edge_weights=None,
                 regions=None, nodes=None, plot_lr=True, coord_range=None, spatial_res=None,
                 node_dist_constant=0.1, k_nearest=None):
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
        self.nodes = torch.from_numpy(nodes).to(self.device).float() if nodes is not None else None
        self.coord_range = coord_range
        self.spatial_res = spatial_res
        self.node_dist_constant = node_dist_constant
        self.k_nearest = k_nearest
        self.edge_weights = edge_weights

        if node2cell is not None:
            self.node2cell = {}
            for i, arr in node2cell.items():
                self.node2cell[i] = torch.from_numpy(arr).float().to(device)

        self.criterion_dict = {
            "MSE": nn.MSELoss(),
            "BCE": nn.BCELoss(),
            "BCElogit": nn.BCEWithLogitsLoss(),
            "likelihood": self.__likelihood_loss,
            "prob_loss": self.__prob_loss
        }

        self.model_step_preds = {key: [] for key in ["train", "val", "test"]}
        self.model_step_labels = deepcopy(self.model_step_preds)
        self.stats = deepcopy(self.model_step_preds)

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
        best_val_score = 0
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
            with torch.no_grad():
                running_val_loss = self.__step_loop(model=model,
                                                    generator=batch_generator,
                                                    mode='val',
                                                    optimizer=None,
                                                    collect_results=True)

            epoch_time = time.time() - start_time

            running_val_score = self.stats["val"][0]
            message_str = "Epoch: {}, Train_loss: {:.5f}, Val_loss: {:.5f}, Val_Score: {:.5f} Took {:.3f} seconds."
            print(message_str.format(epoch + 1, running_train_loss, running_val_loss, running_val_score, epoch_time))

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
                with torch.no_grad():
                    tr_loss, vl_loss = [self.__step_loop(model=model,
                                                         generator=batch_generator,
                                                         mode=mode,
                                                         optimizer=None,
                                                         collect_results=True) for mode in
                                        ["train", "val"]]

                best_val_score = self.stats["val"][0]
                print("-*-" * 10)
                message_str = "Early exiting from epoch: {}, \nTrain Loss: {:5f}, Val Loss: {:5f}." \
                              "Best Val Score {:.5f} Best Train Score {:.5f}"
                print(message_str.format(best_epoch, tr_loss, vl_loss, best_val_score, self.stats["train"][0]))
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
            plt.savefig(os.path.join(self.save_dir, "lr_curve.png"), dpi=200)
            plt.show()

    def transform(self, model, batch_generator):
        test_loss = self.__step_loop(model=model,
                                     generator=batch_generator,
                                     mode='test',
                                     optimizer=None,
                                     collect_results=True)

        print('Test finished, test loss: {:.5f} test score {:.5f}'.format(test_loss, self.stats["test"][0]))
        self.__save_model(model)
        self.__save_outputs()

    def __step_loop(self, model, generator, mode, optimizer, collect_results=False):
        if collect_results:
            # clear the previous collection if any
            self.model_step_preds[mode] = []
            self.model_step_labels[mode] = []

        running_loss = 0
        for idx, batch in enumerate(generator.generate(mode)):
            print('\r{}:{}/{}'.format(mode, idx, generator.num_iter(mode)), flush=True, end='')
            loss = self.__step(model=model,
                               inputs=batch,
                               mode=mode,
                               optimizer=optimizer,
                               dataset_name=generator.dataset_name,
                               collect_outputs=collect_results)

            running_loss += loss
        running_loss /= (idx + 1)

        if collect_results:
            if self.loss_function == "likelihood":
                pred_dict = self.model_step_preds[mode]
                label_dict = self.model_step_labels[mode]
                stats = get_graph_stats(pred_dict, label_dict, self.coord_range, self.spatial_res)
            else:
                y_pred = np.concatenate(self.model_step_preds[mode])
                y_true = np.concatenate(self.model_step_labels[mode])

                if self.loss_function == "MSE":
                    y_true = (y_true > 0).astype(int)

                # calc f1 score
                pred = bin_pred(pred=y_pred.flatten(), label=y_true.flatten())
                score = f1_score(y_true=y_true.flatten(), y_pred=pred)
                print(f"F1 Score: {score}")
                tn, fn, fp, tp = confusion_matrix(y_true=y_true.flatten(), y_pred=pred).flatten()
                print(f"Confusion Matrix = TN:{tn}, FN:{fn}, FP:{fp}, TP:{tp}")
                acc = accuracy_score(y_true=y_true.flatten(), y_pred=pred)
                print(f"Accuracy = {acc:.4f}")
                stats = score, y_pred, y_true

            self.stats[mode] = stats

        return running_loss

    def __step(self, model, inputs, optimizer, mode, dataset_name, collect_outputs):
        if optimizer:
            optimizer.zero_grad()

        x, y = self.__prep_input(inputs[0]), self.__prep_input(inputs[1])
        if dataset_name == "graph":
            edge_index = inputs[2].to(self.device)
            edge_weight = torch.from_numpy(self.edge_weights).float().to(self.device)
            pred = model.forward(x, edge_index=edge_index, edge_weight=edge_weight)
        else:
            pred = model.forward(x)

        loss, pred = self.__get_loss(pred, y)

        if optimizer is not None:
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(model.parameters(), self.clip)

            # take step in classifier's optimizer
            optimizer.step()

        if collect_outputs:
            self.__collect_outputs(pred, y, mode)

        loss = loss.detach().cpu().numpy()
        print(f"Loss: {loss}")
        return loss

    def __get_loss(self, pred, y):
        if self.loss_function in self.custom_losses:
            loss, pred = self.criterion_dict[self.loss_function](pred=pred, y=y)
        else:
            loss = self.criterion_dict[self.loss_function](pred, y)
        return loss, pred

    def __prob_loss(self, pred, y):
        # pred_mu = pred[0]
        # plt.figure()
        # plt.scatter(pred[0].detach().cpu().numpy()[0, :, 0], pred[0].detach().cpu().numpy()[0, :, 1])
        # # plt.scatter(y[0].detach().cpu().numpy()[:, 0], y[0].detach().cpu().numpy()[:, 1])
        # plt.xlim(0, 1)
        # plt.ylim(0, 1)
        # plt.show()
        batch_prob = get_log_like(pred, node2cell=self.node2cell)
        p1 = y * batch_prob
        p2 = (1 - y) * torch.log((1 - batch_prob.exp()).clip(min=1e-5))
        loss = - (p1 + p2).mean()

        # grid_pred = inverse_label(batch_prob.exp().detach().cpu(), self.spatial_res, self.regions)
        # plt.figure()
        # plt.imshow(grid_pred[0])
        # plt.show()

        # dist_nodes = torch.sqrt(torch.sum((pred_mu - self.nodes) ** 2))
        # loss += self.node_dist_constant * dist_nodes

        # y_pred = bin_pred(pred=batch_prob.detach().cpu().numpy().flatten(),
        #                   label=y.detach().cpu().numpy().flatten())
        # f1 = f1_score(y_true=y.detach().cpu().numpy().flatten(), y_pred=y_pred)
        # print(f"F1 Score: {f1}")
        return loss, batch_prob.exp()

    def __likelihood_loss(self, pred, y):
        pred_mu, pred_sigma = pred
        # plt.figure()
        # plt.scatter(pred_mu.detach().cpu().numpy()[0, :, 0], pred_mu.detach().cpu().numpy()[0, :, 1])
        # events = np.concatenate([v.detach().cpu().numpy() for v in y[0].values()])
        # plt.scatter(events[:, 0], events[:, 1])
        # plt.xlim(0, 1)
        # plt.ylim(0, 1)
        # plt.show()
        total_loss = torch.tensor(0).to(self.device).float()
        counter = 0
        batch_dists = []
        for i in range(pred_mu.shape[0]):
            dists = []
            for j in range(pred_mu.shape[1]):
                mu = pred_mu[i, j]
                sigma = torch.eye(2).to(self.device) * pred_sigma[i, j]
                m = MultivariateNormal(mu.T, sigma)
                dists.append(m)
            batch_dists.append(dists)

            label_dict = y[i]
            for key, val in label_dict.items():
                log_likes = dists[int(key)].log_prob(val)
                total_loss += -torch.sum(log_likes)
                counter += len(val)

        total_loss /= counter
        dist_nodes = torch.sqrt(torch.sum((pred_mu - self.nodes) ** 2))
        total_loss += self.node_dist_constant * dist_nodes

        return total_loss, pred

    def __prep_input(self, x):
        if isinstance(x, list):
            if isinstance(x[0], dict):
                x_dev = []
                for i in range(len(x)):
                    x_dev.append({key: val.float().to(self.device) for key, val in x[i].items()})
                x = x_dev
            else:
                x = [x[i].float().to(self.device) for i in range(len(x))]
        else:
            x = x.float().to(self.device)
        return x

    def __collect_outputs(self, pred, y, mode):
        def detach_all(x):
            if isinstance(x, list) or isinstance(x, tuple):
                if isinstance(x[0], dict):
                    x_dev = []
                    for i in range(len(x)):
                        x_dev.append({key: val.detach().cpu().numpy() for key, val in x[i].items()})
                    x = x_dev
                else:
                    x = [x[i].detach().cpu().numpy() for i in range(len(x))]
            else:
                x = x.detach().cpu().numpy()
            return x

        self.model_step_preds[mode].append(detach_all(pred))
        self.model_step_labels[mode].append(detach_all(y))

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
