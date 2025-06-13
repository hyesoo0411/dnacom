"""
engine_Prune_wQAT101.py: Pruning => 가중치만 -1 0 1로 양자화 
"""

from typing import Dict, List, Tuple

import torch

from tqdm.auto import tqdm

import os

import numpy as np

import matplotlib.pyplot as plt

import matplotlib

matplotlib.use("Agg")


def dynamic_threshold(weights, percentage=0.05):
    """
    Calculates a dynamic threshold based on the weight distribution.
    Uses the absolute values of the weights and computes the threshold
    as the value corresponding to the top `percentage` of absolute weights.

    Args:
        weights: The model's weights (torch.Tensor).
        percentage: The percentage of absolute weights to use for threshold.

    Returns:
        threshold: A dynamically computed threshold value.
    """
    abs_weights = torch.abs(weights)  # Take absolute values of the weights
    threshold = torch.quantile(abs_weights, 1 - percentage)  # Get the quantile based on percentage
    return threshold


def ternary_quantize(weights, threshold=None, percentage=0.05):
    """
    Applies ternary quantization to the given weights.
    Maps values to {-1, 0, 1}.

    Args:
        weights: The model's weights (torch.Tensor).
        threshold: A value to determine the threshold for quantization.
                   Values above +threshold are mapped to 1,
                   values below -threshold are mapped to -1,
                   values between are mapped to 0.

    Returns:
        Quantized weights (torch.Tensor).
    """
    if threshold is None:
        # If no threshold is provided, calculate dynamic threshold
        threshold = dynamic_threshold(weights, percentage=0.05)

    quantized_weights = torch.zeros_like(weights)  # Initialize tensor of zeros with same shape
    quantized_weights[weights > threshold] = 1  # Values above threshold become 1
    quantized_weights[weights < -threshold] = -1  # Values below negative threshold become -1
    return quantized_weights


def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    apply_ternary_quantization: bool,
) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
    """
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)
        # Ensure target labels are in float format for BCEWithLogitsLoss
        y = y.float().unsqueeze(1)

        # 1. Forward pass (Apply ternary quantization during forward pass)
        if apply_ternary_quantization is True:
            with torch.no_grad():  # Apply quantization for forward pass only
                quantized_weights = {}  # Store original weights temporarily
                for name, param in model.named_parameters():
                    if "weight" in name:
                        quantized_weights[name] = param.data.clone()  # Store original weights
                        param.data = ternary_quantize(param.data, percentage=0.05)  # Apply ternary quantization

        # 1. Forward pass
        y_pred = model(X)

        # Restore original weights for backward pass
        if apply_ternary_quantization is True:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if "weight" in name:
                        param.data = quantized_weights[name]  # Restore original weights

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)  # Ensure y has the correct shape (batch_size, 1)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = (y_pred >= 0.5).long()
        # print(f"y_pred_class:{y_pred_class}")
        # print(f"y.long:{y.long()}")
        train_acc += (y_pred_class == y.long()).sum().item() / len(y_pred_class)

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    apply_ternary_quantization: bool,
) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:

    (0.0223, 0.8985)
    """

    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # # Store incorrect predictions
    # incorrect_images = []
    # incorrect_labels = []
    # incorrect_preds = []

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)
            # Ensure target labels are in float format
            y = y.float().unsqueeze(1)

            # 1. Forward pass (Apply ternary quantization during forward pass)
            if apply_ternary_quantization is True:
                with torch.no_grad():  # Apply quantization for forward pass only
                    quantized_weights = {}  # Store original weights temporarily
                    for name, param in model.named_parameters():
                        if "weight" in name:
                            quantized_weights[name] = param.data.clone()  # Store original weights
                            param.data = ternary_quantize(param.data, percentage=0.05)  # Apply ternary quantization

            # 1. Forward pass
            test_pred_logits = model(X)

            # Restore original weights for backward pass
            if apply_ternary_quantization is True:
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if "weight" in name:
                            param.data = quantized_weights[name]  # Restore original weights

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = (test_pred_logits >= 0.5).long()
            test_acc += (test_pred_labels == y.long()).sum().item() / len(test_pred_labels)

            # # Store incorrect predictions
            # incorrect_batch_mask = test_pred_labels != y
            # if incorrect_batch_mask.sum() > 0:
            #     incorrect_images.append(X[incorrect_batch_mask])
            #     incorrect_labels.append(y[incorrect_batch_mask])
            #     incorrect_preds.append(test_pred_labels[incorrect_batch_mask])

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)

    # If there are any incorrect predictions, visualize them
    # if len(incorrect_images) > 0:
    #     visualize_incorrect_predictions(
    #         incorrect_images, incorrect_labels, incorrect_preds, dataloader.dataset.classes
    #     )

    return test_loss, test_acc


def visualize_incorrect_predictions(
    incorrect_images, incorrect_labels, incorrect_preds, class_names, save_dir="/home/hyesoo/MNIST/Incorrect/Incorrect"
):
    """Visualizes incorrect predictions by saving the images with labels."""
    # Ensure save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    incorrect_images = torch.cat(incorrect_images, dim=0).cpu()
    incorrect_labels = torch.cat(incorrect_labels, dim=0).cpu()
    incorrect_preds = torch.cat(incorrect_preds, dim=0).cpu()

    # Save up to 5 incorrect predictions
    num_images = min(len(incorrect_images), 5)
    for i in range(num_images):
        img = incorrect_images[i] / 2 + 0.5  # Denormalize (if necessary)
        npimg = img.numpy()

        # Check if the image is grayscale (1 channel) and convert to RGB (3 channels)
        if npimg.shape[0] == 1:  # If the image has 1 channel
            npimg = np.repeat(npimg, 3, axis=0)  # Repeat the channel 3 times to make it RGB

        # Transpose to (H, W, C) format for saving
        npimg = np.transpose(npimg, (1, 2, 0))

        # Create a filename based on the true and predicted class
        true_label = class_names[incorrect_labels[i]]
        pred_label = class_names[incorrect_preds[i]]
        filename = f"{i}_true_{true_label}_pred_{pred_label}.png"
        filepath = os.path.join(save_dir, filename)

        # Create a plot for saving with title
        plt.figure()
        plt.imshow(npimg)
        plt.title(f"True: {true_label}, Pred: {pred_label}")  # Add the title
        # plt.axis("off")  # Turn off axis for cleaner image

        # Save the image with title
        plt.savefig(filepath, bbox_inches="tight")
        plt.close()  # Close the figure to save memory

        print(f"Saved incorrect prediction: {filepath}")


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    device: torch.device,
) -> Dict[str, List[float]]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for
    each epoch.
    In the form: {train_loss: [...],
                train_acc: [...],
                test_loss: [...],
                test_acc: [...]}
    For example if training for epochs=2:
                {train_loss: [2.0616, 1.0537],
                train_acc: [0.3945, 0.3945],
                test_loss: [1.2641, 1.5706],
                test_acc: [0.3400, 0.2973]}
    """
    # Create empty results dictionary
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(
            model=model, dataloader=train_dataloader, loss_fn=loss_fn, optimizer=optimizer, device=device
        )
        test_loss, test_acc = test_step(model=model, dataloader=test_dataloader, loss_fn=loss_fn, device=device)

        # Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # Return the filled results at the end of the epochs
    return results


def train_qat(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    device: torch.device,
    apply_ternary_quantization: bool,
) -> Dict[str, List[float]]:

    # Create empty results dictionary
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    # Loop through training and testing steps for a number of epochs
    for epoch in range(epochs):
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            apply_ternary_quantization=apply_ternary_quantization,
        )
        test_loss, test_acc = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device,
            apply_ternary_quantization=apply_ternary_quantization,
        )
        # Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        # 마지막에 양자화로 저장하기 위해 진행
        if apply_ternary_quantization and epoch == (epochs - 1):
            for name, param in model.named_parameters():
                if "weight" in name:
                    param.data = ternary_quantize(param.data, percentage=0.05)  # Apply ternary quantization

    return results
    # return model
