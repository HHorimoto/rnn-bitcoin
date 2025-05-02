import torch
import torch.nn as nn
import torch.optim as optim

import sys
import os
import yaml
import random
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from PIL import Image

from sklearn.metrics import accuracy_score

from src.utils.seeds import fix_seed
from src.data.dataset import create_dataset
from src.models.models import Net
from src.models.coachs import Coach
from src.visualization.visualize import plot, plot_preds

def main():

    with open('config.yaml') as file:
        config_file = yaml.safe_load(file)
    print(config_file)

    ROOT = config_file['config']['root']
    NUM_EPOCH = config_file['config']['num_epoch']
    BATCH_SIZE = config_file['config']['batch_size']
    LR = config_file['config']['learning_rate']
    RNN_NAME = config_file['config']['rnn_name']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    time_window = 10
    train_loader, test_loader = create_dataset(ROOT, BATCH_SIZE, time_window=time_window)

    hidden_size = 32
    net = Net(rnn_name=RNN_NAME, in_size=4, out_size=2, hidden_size=hidden_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=LR)

    coach = Coach(net, train_loader, test_loader, criterion, optimizer, device, NUM_EPOCH, n_hidden=hidden_size)
    coach.train_test()

    plot({"train": coach.train_loss, "test": coach.test_loss}, "loss")

    plot_preds(coach.net, test_loader, coach.n_hidden, device, RNN_NAME)
    
if __name__ == "__main__":
    fix_seed()
    main()