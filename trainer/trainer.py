import os
import time
import pickle as pkl
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import average_precision_score, f1_score, confusion_matrix

from helpers.graph_helper import get_probs


class Trainer:
    def __init__(self, num_epochs, early_stop_tolerance, clip, optimizer, loss_function,
                 learning_rate, weight_decay, momentum, device, save_dir, node2cell=None, regions=None):
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
        self.custom_losses = ["prob_loss"]
        if node2cell is not None:
            self.node2cell = {}
            for i, arr in node2cell.items():
                self.node2cell[i] = torch.from_numpy(arr).float().to(device)
        self.regions = regions

        self.criterion_dict = {
            "MSE": nn.MSELoss(),
            "BCE": nn.BCELoss(),
            "prob_loss": self.__prob_loss
        }

        self.running_statistics = {
            "train_loss": [],
            "val_loss": [],
            "train_metrics": [],
            "validation_metrics": [],
            "eval_loss": 0,
            "eval_metric": {},
            "test_loss": 0,
            "test_metric": {}
        }

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

        tolerance = 0
        best_epoch = 0
        best_val_loss = 1e6
        best_dict = model.state_dict()
        for epoch in range(self.num_epochs):
            # train and validation loop
            start_time = time.time()

            # train
            train_loss, train_metrics = self.__step_loop(model=model,
                                                         generator=batch_generator,
                                                         mode='train',
                                                         optimizer=optimizer)

            # validation
            val_loss, val_metrics = self.__step_loop(model=model,
                                                     generator=batch_generator,
                                                     mode='val',
                                                     optimizer=None)

            epoch_time = time.time() - start_time

            message_str = "\nEpoch: {}, Train_loss: {:.5f}, Validation_loss: {:.5f}, Took {:.3f} seconds."
            print(message_str.format(epoch + 1, train_loss, val_loss, epoch_time))
            self.print_metrics(train_metrics, metric_name="Train")
            self.print_metrics(val_metrics, metric_name="Validation")

            # save the losses
            self.running_statistics["train_loss"].append(train_loss)
            self.running_statistics["val_loss"].append(val_loss)
            self.running_statistics["train_metrics"].append(train_metrics)
            self.running_statistics["validation_metrics"].append(val_metrics)

            # checkpoint
            self.__save_progress(model)

            if val_loss < best_val_loss:
                best_epoch = epoch + 1
                best_val_loss = val_loss
                best_dict = deepcopy(model.state_dict())
                tolerance = 0
            else:
                tolerance += 1

            if tolerance > self.tolerance or epoch == self.num_epochs - 1:
                model.load_state_dict(best_dict)

                eval_loss, eval_metrics = self.__step_loop(model=model,
                                                           generator=batch_generator,
                                                           mode='train_val',
                                                           optimizer=None)
                # save eval stats
                self.running_statistics["eval_loss"] = eval_loss
                self.running_statistics["eval_metric"] = eval_metrics
                # checkpoint
                self.__save_progress(model)
                break

            torch.cuda.empty_cache()

        message_str = "Early exiting from epoch: {}, Eval Loss: {:.5f}."
        print(message_str.format(best_epoch, self.running_statistics["eval_loss"]))
        self.print_metrics(self.running_statistics["eval_metric"], metric_name="Evaluation")

    def transform(self, model, batch_generator):
        test_loss, test_metrics = self.__step_loop(model=model,
                                                   generator=batch_generator,
                                                   mode='test',
                                                   optimizer=None)
        print('Test finished, test loss: {:.5f}'.format(test_loss))
        self.print_metrics(test_metrics, metric_name="Test")
        self.running_statistics["test_loss"] = test_loss
        self.running_statistics["test_metrics"] = test_metrics
        self.__save_progress(model)

    def __step_loop(self, model, generator, mode, optimizer):
        if mode in ["test", "val", "train_val"]:
            step_fun = self.__val_step
        else:
            step_fun = self.__train_step
        idx = 0
        running_loss = 0
        running_stats = {}
        for idx, batch in enumerate(generator.generate(mode)):
            print('\r{}:{}/{}'.format(mode, idx, generator.num_iter(mode)),
                  flush=True, end='')
            if generator.dataset_name == "graph":
                x, y, edge_index = batch
                x, y = [self.__prep_input(i) for i in [x, y]]
                inputs = [x, y, edge_index.to(self.device)]
            else:
                x, y = batch
                x, y = [self.__prep_input(i) for i in [x, y]]
                inputs = [x, y]
            loss, metrics = step_fun(model=model,
                                     inputs=inputs,
                                     optimizer=optimizer)

            running_loss += loss
            running_stats = self.store_statistics(running_stats, metrics)
        running_loss /= (idx + 1)
        running_stats = {key: val / (idx + 1) for key, val in running_stats.items()}
        return running_loss, running_stats

    def __train_step(self, model, inputs, optimizer):
        if optimizer:
            optimizer.zero_grad()

        if len(inputs) == 3:
            x, y, edge_index = inputs
            pred = model.forward(x, edge_index)
        else:
            x, y = inputs
            pred = model.forward(x)

        loss, pred = self.__get_loss(pred, y)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(model.parameters(), self.clip)

        # take step in classifier's optimizer
        optimizer.step()

        loss = loss.detach().cpu().numpy()
        metrics = self.calculate_metrics(pred[0].detach().cpu().numpy(),
                                         y[0].detach().cpu().numpy())

        print(f"Loss: {loss}, AP: {metrics['AP']:.5f} F1: {metrics['f1']:.5f}")
        return loss, metrics

    def __val_step(self, model, inputs, optimizer):
        if len(inputs) == 3:
            x, y, edge_index = inputs
            pred = model.forward(x, edge_index)
        else:
            x, y = inputs
            pred = model.forward(x)
        loss, pred = self.__get_loss(pred, y)
        metrics = self.calculate_metrics(pred[0].detach().cpu().numpy(),
                                         y[0].detach().cpu().numpy())
        return loss.detach().cpu().numpy(), metrics

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

    def __prep_input(self, x):
        x = x.float().to(self.device)
        return x

    def __save_progress(self, model):
        stats_path = os.path.join(self.save_dir, "statistics.pkl")
        model_path = os.path.join(self.save_dir, "model.pkl")
        for path, obj in zip([stats_path, model_path], [self.running_statistics, model]):
            with open(path, "wb") as f:
                pkl.dump(obj, f)

    @staticmethod
    def calculate_metrics(pred, label):
        pred, label = pred.flatten(), label.flatten()
        ap = average_precision_score(y_true=label, y_score=pred)

        thresholds = np.linspace(pred.min(), pred.max(), 100)
        f1_list = []
        for thr in thresholds:
            bin_pred = (pred >= thr).astype(int)
            f1_list.append(f1_score(label, bin_pred))
        f1_arr = np.array(f1_list)
        best_threshold = thresholds[np.argmax(f1_arr)]
        best_f1 = np.max(f1_arr)

        bin_pred = (pred >= best_threshold).astype(int)
        tn, fn, fp, tp = confusion_matrix(y_true=label, y_pred=bin_pred).flatten()

        metrics = {
            "AP": ap,
            "f1": best_f1,
            "tn": tn,
            "fn": fn,
            "fp": fp,
            "tp": tp
        }

        return metrics

    @staticmethod
    def store_statistics(running, new_metric):
        if not running:  # empty dict
            running = new_metric
        else:
            for key, val in new_metric.items():
                running[key] += val
        return running

    @staticmethod
    def print_metrics(in_metric, metric_name):
        print_msg = "{} metrics: AP:{:.5f}, F1:{:.5f}, " \
                    "Confusion_matrix: [TN:{:.0f}, FN:{:.0f}, FP:{:.0f}, TP:{:.0f}]"
        print(print_msg.format(metric_name, *in_metric.values()))
