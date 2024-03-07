"""LSTM model module"""

import logging
import pathlib
import time
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data

log = logging.getLogger("electric-generation-forecasting")  # Get logger instance.


class SingleStepLSTMForecaster(nn.Module):

    def __init__(
        self,
        n_features: int,
        n_hidden: int,
        n_outputs: int,
        n_lstm_layers: Optional[int] = 1,
        device: Optional[torch.device] = None,
        activation: Optional[bool] = False,
    ):
        """
        Construct an LSTM RNN.

        Parameters
        ----------
        n_features : int
            The number of input features (1 for univariate forecasting).
        n_hidden : int
            The number of neurons in each hidden layer.
        n_outputs : int
            The number of outputs to predict for each training example.
        n_lstm_layers : int
            The number of LSTM layers.
        device : torch.device, optional
            The device to use.
        activation : bool, optional
            Whether to use activation (True) or not (False). Default is False.
        """
        super().__init__()
        self.device = device  # Set option for device selection.

        self.n_lstm_layers = n_lstm_layers
        self.n_hidden = n_hidden
        self.hidden = None  # Initialize variable for hidden state and cell state.

        # LSTM part
        self.lstm = nn.LSTM(
            input_size=n_features,  # The number of expected features in the input x.
            hidden_size=n_hidden,  # The number of features in the hidden state h.
            num_layers=n_lstm_layers,  # The number of recurrent layers in the network.
            batch_first=True,  # Data is transformed in this way
        )
        # Fully connected part with one layer to transform LSTM output to right output dimension.
        fc = [
            nn.Linear(
                in_features=n_hidden, out_features=n_outputs  # size of input
            )  # size of output
        ]
        # Optionally use Tanh activation afterward as its value range corresponds with the min-max scaling range.
        if activation:
            fc.append(nn.Tanh())

        self.fc = nn.Sequential(*fc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input batch

        Returns
        -------
        torch.Tensor
            The model's prediction for this input batch.
        """
        # Initialize hidden state and cell state.
        hidden_state = torch.zeros(self.n_lstm_layers, x.shape[0], self.n_hidden)
        cell_state = torch.zeros(self.n_lstm_layers, x.shape[0], self.n_hidden)

        # Move hidden state and cell state to device.
        hidden_state = hidden_state.to(self.device)
        cell_state = cell_state.to(self.device)

        self.hidden = (hidden_state, cell_state)

        # -------- Forward pass --------
        # Pass through LSTM part.
        x, (h, c) = self.lstm(x, self.hidden)  # LSTM
        # Pass through fully connected part and return prediction.
        return self.fc(h)

    def predict_n_steps(
        self,
        input_data: torch.Tensor,
        n: int,
        target_date_time: pd.DatetimeIndex,
        headers: pd.Index,
    ) -> pd.DataFrame:
        """
        Predict n steps based on previous predictions using the trained single-step LSTM forecaster.

        Parameters
        ----------
        input_data : torch.Tensor
            The initial input data for prediction, shaped as (batch_size, seq_length, num_features).
        n : int
            The number of steps to predict.
        target_date_time : pandas.DatetimeIndex
            The datetime index of the given sample's corresponding target.
        headers : pandas.Index
            The columns in the original dataframe.

        Returns
        -------
        pandas.DataFrame
            The predicted values for the next n steps.
        """
        self.eval()  # Set the model to evaluation mode.
        # Ensure input_data is formatted as a batch.
        samples, labels = input_data
        input_samples = samples.unsqueeze(0)
        predictions = []  # List to store predictions.
        actual_index = pd.DatetimeIndex([], freq=target_date_time.freq)
        log.debug(f"Initial input sample has shape {input_samples.shape}")
        # Loop for n steps.
        for _ in range(n):
            with torch.no_grad():  # Turn off gradient tracking.
                # Predict next step using the current input.
                prediction = self(input_samples)
                log.debug(f"Prediction has shape {prediction.shape}.")

            # Store prediction.
            predictions.append(prediction)
            log.debug(
                f"Current target datetime index is {target_date_time}.\n"
                f"Current input sample has shape {input_samples.shape}."
            )
            # Update `input_data` for the next step prediction.
            # Shift the sequence one step forward by discarding the first time step
            # and appending the predicted value at the end.
            input_samples = torch.cat([input_samples[:, 1:, :], prediction], dim=1)
            actual_index = actual_index.append(target_date_time)
            target_date_time += target_date_time.freq
        # Concatenate predictions along the sequence length dimension.
        return pd.DataFrame(
            torch.cat(predictions, dim=1).squeeze(0).numpy(),
            columns=headers,
            index=actual_index,
        )

    def predict_next_step_from_dataloader(
        self,
        test_loader: data.DataLoader,
        date_time_index: pd.Index,
        headers: pd.Index,
        device: torch.device,
    ) -> pd.DataFrame:
        """
        Predict the next time step for each sample in the test dataloader using the trained single-step LSTM forecaster.

        Parameters
        ----------
        test_loader : torch.utils.data.DataLoader
            The dataloader containing the test dataset.
        date_time_index : pandas.Index
            The datetime index of the targets.
        headers : pandas.Index
            The columns in the original dataframe.
        device : torch.device
            The device to use.

        Returns
        -------
        pd.DataFrame
            The predicted values for the next time step for each sample in the test dataset.

        Raises
        ------
        ValueError
            If target date time index and dataloader do not contain the same number of samples.
        """
        if len(date_time_index) != len(test_loader):
            raise ValueError(
                "Target date time index and data loader must have the same length, "
                "i.e., must contain the same number of samples."
            )
        self.eval()  # Set the model to evaluation mode.
        predictions = []  # List to store predictions.
        actual_index = pd.DatetimeIndex([], freq=date_time_index[0].freq)
        with torch.no_grad():  # Turn off gradient tracking.
            for time_point, (x, _) in zip(
                date_time_index, test_loader
            ):  # Loop over samples in batched train data.
                x = x.to(device)  # Move sample to device.
                prediction = self(x)
                predictions.append(prediction)
                actual_index = actual_index.append(time_point)

        # Concatenate predictions along the batch dimension
        predictions = torch.cat(predictions, dim=0).squeeze().cpu().numpy()
        # Create DataFrame with predictions and date time index
        return pd.DataFrame(
            predictions,
            columns=headers,
            index=actual_index,
        )


class EarlyStopping:
    """Early stopping to terminate training when validation loss stops improving."""

    def __init__(
        self, patience: int = 10, delta: float = 0, verbose: bool = False
    ) -> None:
        """
        Parameters
        ----------
        patience : int, optional
            Number of epochs to wait for improvement before terminating the training. Default is 10.
        delta : float, optional
            Minimum change in the monitored quantity to qualify as an improvement. Default is 0.
        verbose : bool, optional
            If True, prints a message for each patience increment. Default is False.
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss: Optional[float] = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        """
        Check if the validation loss has stopped improving.

        Parameters
        ----------
        val_loss : float
            The validation loss at the current epoch.

        Returns
        -------
        bool
            True if training should be stopped early, False otherwise.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            log.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

        return self.early_stop


def train_model(
    model: SingleStepLSTMForecaster,
    device: torch.device,
    n_epochs: int,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.modules.loss._Loss,
    train_loader: data.DataLoader,
    valid_loader: data.DataLoader,
    path: Optional[pathlib.Path] = None,
    logging_interval: Optional[int] = 10,
    scheduler: Optional = None,
) -> Tuple[SingleStepLSTMForecaster, torch.Tensor, torch.Tensor]:
    """
    Train the model.
    Params
    ------
    model : torch.nn.Module
        The model to train.
    device : torch.device
        The device to train on.
    n_epochs : int
        The number of epochs to train.
    optimizer : torch.optim.Optimizer
        The optimizer to use.
    criterion : torch.nn.modules.loss._Loss
        The loss function to use.
    train_loader : torch.utils.data.DataLoader
        The training dataloader.
    valid_loader : torch.utils.data.DataLoader
        The validation dataloader
    path : pathlib.Path
        A path for saving model checkpoints to.
    logging_interval : int
        The logging output interval.
    scheduler : torch.optim.lr_scheduler.<scheduler>, optional
        A learning-rate scheduler.
    """
    log.info("Start training.")
    start = time.perf_counter()  # Measure overall training time.
    train_losses, valid_losses = (
        [],
        [],
    )  # Initialize history lists for training and validation losses.

    early_stopping = EarlyStopping(patience=10, verbose=True)

    for epoch in range(n_epochs):  # Loop over epochs.
        start_epoch = time.perf_counter()  # Measure training time.
        train_loss, valid_loss = (
            0.0,
            0.0,
        )  # Initialize train and validation losses for this epoch.
        model.train()  # Set model to training mode
        # Layers like dropout behave differently on train and test procedures.
        for batch_idx, (x, y) in enumerate(
            train_loader
        ):  # Loop over mini batches in batched train data.
            x = x.to(
                device
            )  # Move samples consisting of `n_features` features to device.
            y = y.squeeze().to(device)  # Move targets to device.
            predictions = model(
                x
            ).squeeze()  # Forward pass: Apply model to samples to calculate output.
            loss = criterion(predictions, y)  # Compute batch loss.
            train_loss += (
                loss.item()
            )  # Add current batch loss to calculate average train loss of this epoch later on.
            optimizer.zero_grad()  # Zero out gradients from former step.
            loss.backward()  # Backward pass: Compute gradients of loss wrt weights with backprop.
            optimizer.step()  # Perform single optimization step to update model parameters.
        epoch_loss = train_loss / len(
            train_loader
        )  # Calculate average train loss for this epoch.
        train_losses.append(epoch_loss)  # Append train loss to history.

        model.eval()  # Set model to evaluation mode.
        # Loop over validation dataset
        for x, y in valid_loader:  # Loop over mini batches in batched validation data.
            with torch.no_grad():  # Disable gradient calculation to reduce memory consumption during inference.
                x, y = x.to(device), y.squeeze().to(device)  # Move inputs to device.
                predictions = model(x).squeeze()  # Forward pass
                error = criterion(predictions, y)  # Calculate batch loss.
            valid_loss += error.item()
        valid_loss /= len(valid_loader)  # Calculate average valid loss.
        valid_losses.append(valid_loss)  # Append valid loss to history.
        elapsed_epoch = (
            time.perf_counter() - start_epoch
        )  # Measure training time per epoch.
        if epoch % logging_interval == 0:
            log.info(
                f"Epoch {epoch} took {elapsed_epoch:.2f} s: Train loss is {epoch_loss}, "
                f"validation loss is {valid_loss}."
            )

        if early_stopping(valid_loss):
            log.info(
                f"Validation loss has not improved for {early_stopping.patience} epochs. Stopping early."
            )
            break

        if scheduler is not None:  # Adapt learning rate.
            scheduler.step(valid_loss)

    log.info(f"Overall training time: {(time.perf_counter() - start) / 60} min")
    log.info(f"Save model and optimizer state dicts to disk for later inference.")
    p = pathlib.Path(path)
    p.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), p / pathlib.Path(f"model.pt"))
    torch.save(optimizer.state_dict(), p / pathlib.Path(f"optim.pt"))
    return model, torch.tensor(train_losses), torch.tensor(valid_losses)


def plot_losses(
    train_losses: torch.Tensor,
    valid_losses: torch.Tensor,
    logarithmic: Optional[bool] = True,
) -> None:
    """
    Plot training and validation loss after each epoch.

    Parameters
    ----------
    train_losses : torch.Tensor
        The training losses after each epoch.
    valid_losses : torch.Tensor
        The validation losses after each epoch
    logarithmic : bool, optional
        Whether to use a log scale for the losses (True) or not (False.)
        Default is True.
    """
    if logarithmic:
        plt.semilogy(train_losses, label="Training")
        plt.semilogy(valid_losses, label="Validation")
    else:
        plt.plot(train_losses, label="Training")
        plt.plot(valid_losses, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.legend()
    plt.show()
