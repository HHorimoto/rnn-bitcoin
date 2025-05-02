import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, RandomRotation, RandomHorizontalFlip
from PIL import Image
import pathlib
import numpy as np
from os import path
import csv

from src.utils.seeds import worker_init_fn, generator

class BitcoinDataset(torch.utils.data.Dataset):
    def __init__(self, root="./data/BitcoinPricePrediction", train=True, time_window=10, max_price=None):
        super().__init__()

        self.root = root
        self.train = train
        self.time_window = time_window # fixed time frame
        self.max_price = max_price

        if self.train:
            self.csv_file_path = path.join(self.root, 'Training.csv')
        else:
            self.csv_file_path = path.join(self.root, 'Test.csv')

        with open(self.csv_file_path, 'r') as file:
            reader = csv.reader(file)
            csv_data_source = [row for row in reader]

        self.header = csv_data_source[0]
        self.csv_data = csv_data_source[1:]
        self.num_data = len(self.csv_data)

        self.date = [row[0] for row in self.csv_data]

        self.bitcoin_data = torch.zeros([self.num_data, 4], dtype=torch.float32)
        for i, row in enumerate(self.csv_data):
            self.bitcoin_data[i, 0] = float(row[1])
            self.bitcoin_data[i, 1] = float(row[2])
            self.bitcoin_data[i, 2] = float(row[3])
            self.bitcoin_data[i, 3] = float(row[4])
        
        # new to old --> old to new
        self.date.reverse()
        self.bitcoin_data = torch.flipud(self.bitcoin_data)

        # normalization
        if self.max_price is None:
            self.max_price = torch.max(self.bitcoin_data)
        self.bitcoin_data /= self.max_price

    def __getitem__(self, index):
        X = self.bitcoin_data[index:index+self.time_window, :] # (Open, High, Low, Close)
        y = self.bitcoin_data[index+1:index+self.time_window+1, 1:3] # (High, Low)

        return X, y

    def __len__(self):
        return self.num_data - self.time_window

def create_dataset(root, batch_size, time_window=10):
    train_dataset = BitcoinDataset(root=root, train=True, time_window=time_window, max_price=None)
    test_dataset = BitcoinDataset(root=root, train=False, time_window=1, max_price=train_dataset.max_price)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn,)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn,)
    
    datasets = {"train": train_dataset, "test": test_dataset}
    dataloaders = {"train": train_loader, "test": test_loader}
    
    return datasets, dataloaders