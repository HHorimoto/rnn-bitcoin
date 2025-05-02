import torch

import numpy as np
import matplotlib.pyplot as plt

def plot(results: dict, metric: str):
    plt.figure()
    plt.xlabel('epoch')
    plt.ylabel(metric)
    for key, value in results.items():
        plt.plot(value, label=key)
    plt.legend()
    plt.savefig(metric+'.png')

def plot_preds(net, dataloader, n_hidden, device, rnn_name):
    net.eval()
    pred, true = [], []

    hx = torch.zeros(1, n_hidden).to(device)
    cx = torch.zeros(1, n_hidden).to(device)

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            output, hx, cx = net(X[:, 0, :], hx, cx)

            pred.append(output.tolist())
            true.append(y.tolist())
    
    pred_tensor = torch.tensor(pred).squeeze()
    true_tensor = torch.tensor(true).squeeze()

    time_index = list(range(pred_tensor.size(0)))
    print("time index list:", time_index)

    plt.figure()
    plt.plot(time_index, pred_tensor[:, 0], '-b', label='high pred')
    plt.plot(time_index, true_tensor[:, 0], '-r', label='high true')
    plt.plot(time_index, pred_tensor[:, 1], '-c', label='low pred')
    plt.plot(time_index, true_tensor[:, 1], '-y', label='low true')
    plt.xlabel("day")
    plt.ylabel("price")
    plt.title("Prediction Results for Test Data")
    plt.legend()
    plt.savefig('preds.png')
    