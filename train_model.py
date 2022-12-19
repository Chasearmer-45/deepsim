import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd
import wandb

import math
import os

class SimulationDataset(Dataset):
    def __init__(self, step_size, OOD=False):
        if not OOD:
            self.data = pd.read_csv(f"datasets/tellurium_dataset_step_{step_size}.csv")
        else:
            self.data = pd.read_csv(f"datasets/tellurium_OOD_dataset_step_{step_size}.csv")


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.loc[idx]
        X, Y = item[:31], item[31:]
        X, Y = torch.tensor(X.values).float(), torch.tensor(Y.values).float()
        return X, Y

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        hidden_size = 256
        num_hidden_layers = 1
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(31, hidden_size),
            nn.ReLU(),
            *[nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU()) for _ in range(num_hidden_layers - 1)],
            nn.Linear(hidden_size, 5)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

def train(dataloader, model, loss_fn, optimizer, epoch, device):
    # define logging variables
    dataset_size = len(dataloader.dataset)
    n_steps_per_epoch = math.ceil(dataset_size / wandb.config.batch_size)
    example_count = epoch * dataset_size

    # train on each batch in the dataset
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # log metrics to wandb
        example_count += len(X)
        metrics = {"train/train_loss": loss,
                   "train/epoch": (batch + 1 + (n_steps_per_epoch * epoch)) / n_steps_per_epoch,
                   "train/example_ct": example_count}
        if batch + 1 < n_steps_per_epoch:
            wandb.log(metrics)

        # print training progress
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"train loss: {loss:>7f}  [{current:>5d}/{dataset_size:>5d}]")

    return

def validate(dataloader, model, loss_fn, device, OOD=False):
    # define logging variables
    num_batches = len(dataloader)
    val_loss = 0

    # evaluate on validation dataset
    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            val_loss += loss_fn(pred, y).item()

    # log and print val loss
    val_loss /= num_batches
    if not OOD:
        wandb.log({"val/val_loss": val_loss})
        print(f"Avg validation loss: {val_loss:>8f}")
    else:
        wandb.log({"val/OOD_loss": val_loss})
        print(f"Avg out-of-distribution loss: {val_loss:>8f} \n")

    return

def train_model(batch_size, learning_rate, epochs, step_size):

    # split data into train and validation
    simulation_dataset = SimulationDataset(step_size)
    train_size = int(0.5 * len(simulation_dataset))
    valid_size = len(simulation_dataset) - train_size
    training_data, validation_data = torch.utils.data.random_split(simulation_dataset, [train_size, valid_size])

    # get OOD dataset
    OOD_data = SimulationDataset(step_size, OOD=True)

    # Create data loaders
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)
    OOD_dataloader = DataLoader(OOD_data, batch_size=batch_size, shuffle=False)

    for X, y in valid_dataloader:
        print(f"Shape of X: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    # get cpu or gpu device for training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # define model
    model = NeuralNetwork().to(device)
    print(model)

    # define loss and optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # train model
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, epoch, device)
        validate(valid_dataloader, model, loss_fn, device)
        validate(OOD_dataloader, model, loss_fn, device, OOD=True)
        wandb.watch(model)

    print("Done!")

    return model

if __name__ == "__main__":
    '''
    Each step size must be run individually by modifying the value of 'step_size' in the wandb.init function below.
    '''

    # initialize WandB
    wandb.init(
        project="de-solver",
        config={
            "learning_rate": 0.001,
            "epochs": 20,
            "batch_size": 32,
            "step_size": 1
        }
    )

    # train model
    model = train_model(**wandb.config)

    # Create the model weights directory, if necessary
    weights_dir = "model_weights/"
    if not os.path.exists(weights_dir):
        os.mkdir(weights_dir)

    # save model weights
    torch.save(model.state_dict(), f"{weights_dir}model_step_{wandb.config.step_size}.p")
