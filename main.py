import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import pandas as pd
import numpy as np
from SDT import SDT
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
import random
from torch.utils.data import TensorDataset, DataLoader
import os
import matplotlib.pyplot as plt
import argparse


def mape(y_test, pred):
    mape = np.mean(np.abs((y_test - pred) / y_test))
    return mape


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, help="Data path for forecast", default="experiment")
    parser.add_argument("--data_path", type=str, help="Data path for forecast")
    parser.add_argument("--label_name", type=str, help="label name like y, target or etc", default="y")
    parser.add_argument("--test_size", type=float, default=0.3, help="Test size as portion")
    parser.add_argument("--depth", type=int, default=3, help="Tree Depth")
    parser.add_argument("--lamda", type=float, default=1e-3, help="coefficient of the regularization term")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--epochs", type=int, default=30, help="The number of training epoch")
    parser.add_argument("--date_column_name", type=str, help="whether there is date index", default=None)

    args = parser.parse_args()

    exp_name = args.exp_name
    data_path = args.data_path
    label_name = args.label_name
    test_size = args.test_size
    date_column_name = args.date_column_name

    depth = args.depth  # tree depth
    lamda = args.lamda  # coefficient of the regularization term
    lr = args.lr  # learning rate
    epochs = args.epochs  # the number of training epochs

    if not os.path.exists("Results"):
        os.makedirs("Results")

    result_folder = f"Results/sdt_arm_{exp_name}"
    result_txt_file = f"sdt_arma_{exp_name}.txt"

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # Parameters
    # the number of input dimensions
    output_dim = 1  # the number of outputs (i.e., # classes on MNIST)
    batch_size = 1  # batch size
    use_cuda = torch.cuda.is_available()  # whether to use GPU

    device = torch.device("cuda" if use_cuda else "cpu")

    # reproducibility
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)

    # Load data

    ##########################

    mu, sigma = 0, 0.1

    df = pd.read_csv(data_path)
    df = df.drop("Unnamed: 0", axis=1)
    if date_column_name is not None:
        df = df.drop(date_column_name, axis=1)
    col_list = list(df.columns)
    col_list.remove(label_name)
    col_list.insert(0, label_name)

    df = df[col_list]

    y = df.loc[:, label_name]
    X = df.iloc[:, 1:]
    data_len = len(X)

    forecast_horizon = int(data_len * test_size)
    e_t = np.random.normal(mu, sigma, len(X))

    X["e"] = e_t
    X["e-1"] = 0
    X["e-2"] = 0
    X["e-3"] = 0

    X_train, X_test = X.iloc[:-forecast_horizon].to_numpy().astype(np.float32), X.iloc[
        -forecast_horizon:
    ].to_numpy().astype(np.float32)
    y_train, y_test = y.iloc[:-forecast_horizon].to_numpy().astype(np.float32), y.iloc[
        -forecast_horizon:
    ].to_numpy().astype(np.float32)

    scaler = MinMaxScaler()
    X_train_arr = scaler.fit_transform(X_train)
    X_test_arr = scaler.transform(X_test)

    y_train_arr = scaler.fit_transform(y_train.reshape(-1, 1))
    y_test_arr = scaler.transform(y_test.reshape(-1, 1))

    train_features = torch.Tensor(X_train_arr).to(device)
    train_targets = torch.Tensor(y_train_arr).to(device)
    test_features = torch.Tensor(X_test_arr).to(device)
    test_targets = torch.Tensor(y_test_arr).to(device)

    train = TensorDataset(train_features, train_targets)
    test = TensorDataset(test_features, test_targets)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    input_dim = train_features.shape[1]
    print(f"input_dim: {input_dim}")
    ##########################

    # Model and Optimizer
    tree = SDT(input_dim, output_dim, depth, lamda, use_cuda)
    tree = tree.to(device)

    optimizer = torch.optim.SGD(tree.parameters(), lr=lr)

    criterion = nn.L1Loss()

    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        print("###############")
        print(f"epoch: {epoch}")
        # Training
        tree.train()
        train_target_list = []
        train_output_list = []
        e_1 = 0
        e_2 = 0
        e_3 = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data[0][-3] = e_1
            data[0][-2] = e_2
            data[0][-1] = e_3

            batch_size = data.size()[0]
            data, target = data.to(device), target.to(device)

            output, penalty = tree.forward(data, is_training_data=True)

            e_3 = e_2
            e_2 = e_1
            e_1 = target.item() - output.item()

            train_output_list.append(output.cpu().detach().numpy())
            train_target_list.append(target.cpu().detach().numpy())

            loss = criterion(output, target)
            loss += penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_target_list = np.array(train_target_list).ravel()
        train_output_list = np.array(train_output_list).ravel()

        train_target_list = scaler.inverse_transform(train_target_list.reshape(-1, 1))
        train_output_list = scaler.inverse_transform(train_output_list.reshape(-1, 1))

        train_mse = mse(train_target_list, train_output_list)
        train_mae = mae(train_target_list, train_output_list)
        train_mape = mape(train_target_list, train_output_list)
        print(f"traim mse {train_mse}")
        print(f"train mae {train_mae}")
        print(f"train mape {train_mape}")
        print("--------")

        train_losses.append((train_mse, train_mae, train_mape))

        tree.eval()
        correct = 0.0
        output_list = []
        target_list = []

        for batch_idx, (data, target) in enumerate(test_loader):
            data[0][-3] = e_1
            data[0][-2] = e_2
            data[0][-1] = e_3

            batch_size = data.size()[0]
            data, target = data.to(device), target.to(device)

            output = tree.forward(data)

            e_3 = e_2
            e_2 = e_1
            e_1 = target.item() - output.item()

            output_list.append(output.cpu().detach().numpy())
            target_list.append(target.cpu().detach().numpy())

        target_list = np.array(target_list).ravel()
        output_list = np.array(output_list).ravel()

        target_list = scaler.inverse_transform(target_list.reshape(-1, 1))
        output_list = scaler.inverse_transform(output_list.reshape(-1, 1))

        test_mse = mse(target_list, output_list)
        test_mae = mae(target_list, output_list)
        test_mape = mape(target_list, output_list)
        print(f"test mse {test_mse}")
        print(f"test mae {test_mae}")
        print(f"test mape {test_mape}")

        test_losses.append((test_mse, test_mae, test_mape))

        print("###############")

    plt.plot(target_list)
    plt.plot(output_list)
    plt.savefig(f"{result_folder}/predictions_{exp_name}.png")
    plt.cla()
    f = open(f"{result_folder}/{result_txt_file}", "a+")
    f.write("########\n")
    f.write(f"M4 index: {exp_name}\n")

    f.write("Train Losses")
    f.write(str(train_losses) + "\n\n")

    f.write("Test Losses")
    f.write(str(test_losses) + "\n\n")
    f.write("########\n")
    f.close()

    test_losses = [l[0] for l in test_losses]
    train_losses = [l[0] for l in train_losses]

    plt.plot(test_losses, label="test")
    plt.plot(train_losses, label="train")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("MSE Losses")
    plt.savefig(f"{result_folder}/SDT_result_{exp_name}.png")
    plt.cla()
