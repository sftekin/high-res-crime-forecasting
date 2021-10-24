import time
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy


class Trainer:
    def __init__(self, num_epochs, early_stop_tolerance, clip, optimizer, loss_function,
                 learning_rate, weight_decay, momentum, device, node2cell=None):
        self.num_epochs = num_epochs
        self.clip = clip
        self.optimizer = optimizer
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.tolerance = early_stop_tolerance
        self.device = torch.device(device)
        self.loss_function = loss_function
        self.custom_losses = ["prob_loss"]
        if node2cell is not None:
            self.node2cell = {}
            for i, arr in node2cell.items():
                self.node2cell[i] = torch.from_numpy(arr).float().to(device)

        self.criterion_dict = {
            "MSE": nn.MSELoss(),
            "BCE": nn.BCELoss(),
            "prob_loss": self.__prob_loss
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

        train_loss = []
        val_loss = []
        tolerance = 0
        best_val_loss = 1e6
        best_epoch = 0
        evaluation_val_loss = best_val_loss
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

            message_str = "\nEpoch: {}, Train_loss: {:.5f}, Validation_loss: {:.5f}, Took {:.3f} seconds."
            print(message_str.format(epoch + 1, running_train_loss, running_val_loss, epoch_time))

            # save the losses
            train_loss.append(running_train_loss)
            val_loss.append(running_val_loss)

            if running_val_loss < best_val_loss:
                best_epoch = epoch + 1
                best_val_loss = running_val_loss
                best_dict = deepcopy(model.state_dict())
                tolerance = 0
            else:
                tolerance += 1

            if tolerance > self.tolerance or epoch == self.num_epochs - 1:
                model.load_state_dict(best_dict)

                evaluation_val_loss = self.__step_loop(model=model,
                                                       generator=batch_generator,
                                                       mode='train_val',
                                                       optimizer=None)

                message_str = "Early exiting from epoch: {}, Validation error: {:.5f}."
                print(message_str.format(best_epoch, evaluation_val_loss))
                break

            torch.cuda.empty_cache()

        print('Train finished, best eval lost: {:.5f}'.format(evaluation_val_loss))
        return train_loss, val_loss, evaluation_val_loss

    def transform(self, model, batch_generator):
        test_loss = self.__step_loop(model=model,
                                     generator=batch_generator,
                                     mode='test',
                                     optimizer=None)
        print('Test finished, test loss: {:.5f}'.format(test_loss))
        return test_loss

    def __step_loop(self, model, generator, mode, optimizer):
        running_loss = 0
        if mode in ['test', 'val']:
            step_fun = self.__val_step
        else:
            step_fun = self.__train_step
        idx = 0
        for idx, (x, y, edge_index) in enumerate(generator.generate(mode)):
            print('\r{}:{}/{}'.format(mode, idx, generator.num_iter(mode)),
                  flush=True, end='')
            x, y = [self.__prep_input(i) for i in [x, y]]
            loss = step_fun(model=model,
                            inputs=[x, y, edge_index.to(self.device)],
                            optimizer=optimizer)

            running_loss += loss
        running_loss /= (idx + 1)

        return running_loss

    def __train_step(self, model, inputs, optimizer):
        x, y, edge_index = inputs
        if optimizer:
            optimizer.zero_grad()
        pred = model.forward(x, edge_index)

        loss = self.__get_loss(pred, y)

        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(model.parameters(), self.clip)

        # take step in classifier's optimizer
        optimizer.step()

        loss = loss.detach().cpu().numpy()
        print(f"  loss: {loss}")

        return loss

    def __val_step(self, model, inputs, optimizer):
        x, y, edge_index = inputs
        pred = model.forward(x, edge_index)

        loss = self.__get_loss(pred, y)

        return loss.detach().cpu().numpy()

    def __get_loss(self, pred, y, **kwargs):
        if self.loss_function in self.custom_losses:
            loss = self.criterion_dict[self.loss_function](pred=pred, y=y, **kwargs)
        else:
            loss = self.criterion_dict[self.loss_function](pred, y)

        return loss

    def __prob_loss(self, pred, y):
        criterion = self.criterion_dict["BCE"]
        pred_mu, pred_sigma, mix_coef = pred
        batch_prob = []
        for batch_id in range(pred_mu.shape[0]):
            prob = []
            for node_id, cell_arr in self.node2cell.items():
                mu1, mu2 = pred_mu[batch_id, node_id]
                sigma1, sigma2 = pred_sigma[batch_id, node_id]

                p1 = self.__calc_prob(cell_arr[:, 0], mu1, sigma1)
                p2 = self.__calc_prob(cell_arr[:, 1], mu2, sigma2)
                prob.append(p1 * p2)
            prob = torch.cat(prob)
            batch_prob.append(prob)
        batch_prob = torch.stack(batch_prob)
        loss = criterion(batch_prob, y)
        return loss

    def __prep_input(self, x):
        x = x.float().to(self.device)
        # # (b, t, m, n, d) -> (b, t, d, m, n)
        # x = x.permute(0, 1, 4, 2, 3)
        return x

    @staticmethod
    def __calc_prob(x, mu, sigma):
        x1 = (x[:, 0] - mu) / (sigma * 1.41)
        x2 = (x[:, 1] - mu) / (sigma * 1.41)
        prob = (torch.erf(x2) - torch.erf(x1)) * 0.5
        return prob

    @staticmethod
    def inverse_label(pred, label_shape, regions):
        batch_size = pred.shape[0]
        grid = torch.zeros(batch_size, *label_shape)
        prev_idx = 0
        for r, c in regions:
            row_count = r[1] - r[0]
            col_count = c[1] - c[0]
            cell_count = row_count * col_count
            grid[:, r[0]:r[1], c[0]:c[1]] = \
                pred[:, prev_idx:prev_idx + cell_count].reshape(-1, row_count, col_count)
            prev_idx += cell_count

        return grid
