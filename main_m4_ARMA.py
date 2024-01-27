"""Training and evaluating a soft decision tree on the MNIST dataset."""

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


# def onehot_coding(target, device, output_dim):
#     """Convert the class labels into one-hot encoded vectors."""
#     target_onehot = torch.FloatTensor(target.size()[0], output_dim).to(device)
#     target_onehot.data.zero_()
#     target_onehot.scatter_(1, target.view(-1, 1), 1.0)
#     return target_onehot
def mape(y_test, pred):
    mape = np.mean(np.abs((y_test - pred) / y_test))
    return mape


type_dic = {"Daily": 500, "Weekly": 251, "Hourly": 414}
forecast_dic = {"Daily": 14, "Weekly": 13, "Hourly": 48}
if __name__ == "__main__":
    m4_type = "Weekly"
    result_folder = f"Results/last_result_soft_ARMA_{m4_type}"
    result_txt_file = f"m4_sdt_arma_{m4_type}.txt"

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # M4 Parameters
    # M4_index = 103
    forecast_horizon = forecast_dic[m4_type]
    # Parameters
    # the number of input dimensions
    output_dim = 1  # the number of outputs (i.e., # classes on MNIST)
    depth = 2  # tree depth
    lamda = 1e-3  # coefficient of the regularization term
    lr = 1e-2  # learning rate
    weight_decaly = 5e-4  # weight decay
    batch_size = 1  # batch size
    epochs = 20  # the number of training epochs
    log_interval = 100  # the number of batches to wait before printing logs
    use_cuda = True  # whether to use GPU

    device = torch.device("cuda" if use_cuda else "cpu")

    # reproducibility
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)
    M4_index = 100 #random.sample(range(type_dic[m4_type]), 50)
    # Load data

    ##########################

    mu, sigma = 0, 0.1
    
    df = pd.read_csv(f"Dataset/Extracted_M4/{M4_index}_M4_{m4_type}.csv")
    df = df.drop("Unnamed: 0", axis=1)
    y = df.loc[:, "y"]
    X = df.iloc[:, 1:]
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
    # torch.autograd.set_detect_anomaly(True)
    for epoch in range(epochs):
        print("###############")
        print(f"epoch: {epoch}")
        # Training
        tree.train()
        train_target_list = []
        train_output_list = []
        e_1 = torch.zeros((1, 1), requires_grad=True)
        e_2 = torch.zeros((1, 1), requires_grad=True)
        e_3 = torch.zeros((1, 1), requires_grad=True)
        for batch_idx, (data, target) in enumerate(train_loader):
            data[0][-3] = e_1
            data[0][-2] = e_2
            data[0][-1] = e_3

            batch_size = data.size()[0]
            data, target = data.to(device), target.to(device)
            # target_onehot = onehot_coding(target, device, output_dim)

            output, penalty = tree.forward(data, is_training_data=True)

            e_3 = e_2
            e_2 = e_1
            e_1 = target - output

            train_output_list.append(output.cpu().detach().numpy())
            train_target_list.append(target.cpu().detach().numpy())

            loss = criterion(output, target)
            loss += penalty

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
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

        # Print training status
        # if batch_idx % log_interval == 0:
        #     pred = output.data.max(1)[1]
        #     correct = pred.eq(target.view(-1).data).sum()

        # msg = "Epoch: {:02d} | Batch: {:03d} | Loss: {:.5f} |" " Correct: {:03d}/{:03d}"
        # print(msg.format(epoch, batch_idx, loss, correct, batch_size))
        # training_loss_list.append(loss.cpu().data.numpy())

        # Evaluating
        tree.eval()
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
            e_1 = target - output

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
    plt.savefig(f"{result_folder}/predictions_{M4_index}.png")
    plt.cla()
    f = open(f"{result_folder}/{result_txt_file}", "a+")
    f.write("########\n")
    f.write(f"M4 index: {M4_index}\n")

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
    plt.savefig(f"{result_folder}/SDT_result_{M4_index}.png")
    plt.cla()
# torch.autograd.set_detect_anomaly(False)
# correct += pred.eq(target.view(-1).data).sum()

# accuracy = 100.0 * float(correct) / len(test_loader.dataset)

# if accuracy > best_testing_acc:
#     best_testing_acc = accuracy

# msg = "\nEpoch: {:02d} | Testing Accuracy: {}/{} ({:.3f}%) |" " Historical Best: {:.3f}%\n"
# print(msg.format(epoch, correct, len(test_loader.dataset), accuracy, best_testing_acc))
# testing_acc_list.append(accuracy)
