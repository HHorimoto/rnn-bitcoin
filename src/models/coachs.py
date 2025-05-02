import torch
import torch.nn as nn

import numpy as np
import time

class Coach:
    def __init__(self, net, train_loader, test_loader, criterion, optimizer, device, num_epoch, n_hidden):
        self.net = net
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.num_epoch = num_epoch
        self.n_hidden = n_hidden

        # store
        self.train_loss, self.test_loss = [], []

    def _train_epoch(self):
        self.net.train()
        dataloader = self.train_loader
        batch_loss = []
        for X, y in dataloader:
            batch_size = X.size(0)
            hx = torch.zeros(batch_size, self.n_hidden).to(self.device)
            cx = torch.zeros(batch_size, self.n_hidden).to(self.device)
            X, y = X.to(self.device), y.to(self.device)

            loss = 0
            time_window = X.size(1)
            for idx_window in range(time_window):
                output, hx, cx = self.net(X[:, idx_window, :], hx, cx)
                loss += self.criterion(output, y[:, idx_window, :])

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_loss.append(loss.item())

        epoch_loss = np.mean(batch_loss)
        return epoch_loss
    
    def _test_epoch(self):
        self.net.eval()
        dataloader = self.test_loader
        batch_loss = []

        hx = torch.zeros(1, self.n_hidden).to(self.device)
        cx = torch.zeros(1, self.n_hidden).to(self.device)
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)

                output, hx, cx = self.net(X[:, 0, :], hx, cx)
                loss = self.criterion(output, y[:, 0, :])

                batch_loss.append(loss.item())

        epoch_loss = np.mean(batch_loss)
        return epoch_loss
    
    def train_test(self):
        start = time.time()
        for epoch in range(self.num_epoch):
            train_epoch_loss = self._train_epoch()
            test_epoch_loss = self._test_epoch()

            print("epoch: ", epoch+1, "/", self.num_epoch)
            print("time: ", time.time()-start)
            print("[train] loss: ", train_epoch_loss)
            print("[test] loss: ", test_epoch_loss)

            self.train_loss.append(train_epoch_loss)
            self.test_loss.append(test_epoch_loss)