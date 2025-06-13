"""
engine_wQAT101_Prune.py: 가중치만 -1 0 1로 scale, round 함수 이용해서 양자화 => Pruning
"""

from typing import Dict, List, Tuple

import torch

from tqdm.auto import tqdm

import os

import numpy as np

import matplotlib.pyplot as plt

import matplotlib

from noise import apply_noise

from utils import custom_quantize, calculate_scale

matplotlib.use("Agg")


# 훈련 과정에서 적용할 QAT 함수
def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    apply_quantization: bool = True,
    levels: List[float] = [-1.0, 0.0, 1.0],
) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch with ternary quantization.

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
        y = y.long()

        # 1. Forward pass (Apply ternary quantization during forward pass)
        if apply_quantization:
            with torch.no_grad():  # Apply quantization for forward pass only
                original_weights = {}
                scales = {}

                # Step 1: Store original weights and apply normalization/quantization in one loop
                for name, param in model.named_parameters():
                    if "weight" in name:
                        original_weights[name] = param.data.clone()  # Save original weights
                        # print(f"original_weight: {original_weights[name][0]}")
                        # Calculate scale and apply ternary quantization
                        scale = calculate_scale(param.data)
                        scales[name] = scale
                        param.data = custom_quantize(param.data, levels, scale)
                        # print(f"Quantized weight: {param.data[0]}")  # Print quantized weights for debugging
        
        # Forward pass through the model up to the linear layer
        X = apply_noise(X, noise_range=(1-model.noise, 1+model.noise))
        linear_output = model.layer_stack[0](X)  # [32, 49]
        linear_output = apply_noise(model.layer_stack[1](linear_output), noise_range=(1-model.noise, 1+model.noise)) # Linear layer
        # print("Shape of linear_output before activation:", linear_output.shape) [32, 200]

        # Apply per-unit normalization on linear_output
        with torch.no_grad():
            # Access the first Linear layer's weights in the layer stack
            weight_matrix = model.layer_stack[1].weight.data
            # print("Weight matrix shape:", weight_matrix.shape) # [200, 49]

            for i in range(weight_matrix.size(0)):  # Iterate over each unit (row) of the weight matrix
                row = weight_matrix[i]
                non_zero_count = (row != 0).sum().item()
                if non_zero_count > 0:
                    linear_output[:, i] = linear_output[:, i] / non_zero_count

        # Pass through the remaining layers (including activation functions)
        linear_output = apply_noise(model.layer_stack[2](linear_output), noise_range=(1-model.noise, 1+model.noise)) # Apply ReLU activation

        y_pred = apply_noise(model.classifier(linear_output), noise_range=(1-model.noise, 1+model.noise))  # Pass through the classifier [32, 10]
        
        # Restore original weights for backward pass
        if apply_quantization:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if "weight" in name:
                        param.data = original_weights[name]  # Restore original weights
                        # if batch == 0:
                        #     print(f"Dequantize: {param.data}, shape: {param.data.size()}")


        # Calculate and accumulate loss
        loss = loss_fn(y_pred, y)
        # print(f"Batch {batch+1}/{len(dataloader)} - Loss: {loss.item():.4f}")  # Print loss for each batch
        train_loss += loss.item()

        # Optimizer zero grad
        optimizer.zero_grad()

        # Loss backward
        loss.backward()

        # Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = y_pred.argmax(dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred_class)

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    apply_quantization: bool = True,
    levels: List[float] = [-1.0, 0.0, 1.0],
) -> Tuple[float, float]:
    """
    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").
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
            y = y.long()

            # 1. Forward pass (Apply ternary quantization during forward pass)
            if apply_quantization:
                with torch.no_grad():  # Apply quantization for forward pass only
                    original_weights = {}
                    scales = {}  # Store scales for dequantization

                    for name, param in model.named_parameters():
                        if "weight" in name:
                            scale = calculate_scale(param.data)  # Calculate scale
                            scales[name] = scale
                            original_weights[name] = param.data.clone()
                            param.data = custom_quantize(param.data, levels, scale)  # Apply ternary quantization

            # Forward pass through the model up to the linear layer
            X = apply_noise(X, noise_range=(1-model.noise, 1+model.noise))
            linear_output = model.layer_stack[0](X)  # Example, assuming first layer is linear (adjust as needed)
            linear_output = apply_noise(model.layer_stack[1](linear_output), noise_range=(1-model.noise, 1+model.noise)) # Linear layer

            # Apply per-unit normalization on linear_output
            with torch.no_grad():
                # Access the first Linear layer's weights in the layer stack
                weight_matrix = model.layer_stack[1].weight.data
                #print("Weight matrix shape:", weight_matrix.shape)

                for i in range(weight_matrix.size(0)):  # Iterate over each unit (row) of the weight matrix
                    row = weight_matrix[i]
                    non_zero_count = (row != 0).sum().item()
                    if non_zero_count > 0:
                        linear_output[:, i] = linear_output[:, i] / non_zero_count

            # Pass through the remaining layers (including activation functions)
            linear_output = apply_noise(model.layer_stack[2](linear_output), noise_range=(1-model.noise, 1+model.noise)) # Apply ReLU activation

            test_pred_logits = apply_noise(model.classifier(linear_output), noise_range=(1-model.noise, 1+model.noise))  # Pass through the classifier

            #test_pred_logits = model(X)

            # Restore original weights for backward pass
            if apply_quantization:
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if "weight" in name:
                            param.data = original_weights[name]  # Restore original weights

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += (test_pred_labels == y).sum().item() / len(test_pred_labels)


    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)

    # If there are any incorrect predictions, visualize them
    # if len(incorrect_images) > 0:
    #     visualize_incorrect_predictions(
    #         incorrect_images, incorrect_labels, incorrect_preds, dataloader.dataset.classes
    #     )

    return test_loss, test_acc


# def visualize_incorrect_predictions(
#     incorrect_images, incorrect_labels, incorrect_preds, class_names, save_dir="/home/hyesoo/DNAcomputing/Incorrect"
# ):
#     """Visualizes incorrect predictions by saving the images with labels."""
#     # Ensure save directory exists
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)

#     incorrect_images = torch.cat(incorrect_images, dim=0).cpu()
#     incorrect_labels = torch.cat(incorrect_labels, dim=0).cpu()
#     incorrect_preds = torch.cat(incorrect_preds, dim=0).cpu()

#     # Save up to 5 incorrect predictions
#     num_images = min(len(incorrect_images), 5)
#     for i in range(num_images):
#         img = incorrect_images[i] / 2 + 0.5  # Denormalize (if necessary)
#         npimg = img.numpy()

#         # Check if the image is grayscale (1 channel) and convert to RGB (3 channels)
#         if npimg.shape[0] == 1:  # If the image has 1 channel
#             npimg = np.repeat(npimg, 3, axis=0)  # Repeat the channel 3 times to make it RGB

#         # Transpose to (H, W, C) format for saving
#         npimg = np.transpose(npimg, (1, 2, 0))

#         # Create a filename based on the true and predicted class
#         true_label = class_names[incorrect_labels[i]]
#         pred_label = class_names[incorrect_preds[i]]
#         filename = f"{i}_true_{true_label}_pred_{pred_label}.png"
#         filepath = os.path.join(save_dir, filename)

#         # Create a plot for saving with title
#         plt.figure()
#         plt.imshow(npimg)
#         plt.title(f"True: {true_label}, Pred: {pred_label}")  # Add the title
#         # plt.axis("off")  # Turn off axis for cleaner image

#         # Save the image with title
#         plt.savefig(filepath, bbox_inches="tight")
#         plt.close()  # Close the figure to save memory

#         print(f"Saved incorrect prediction: {filepath}")


def train_qat(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    device: torch.device,
    apply_quantization: bool = True,
    levels: List[float] = [0, 0.5, 0.7, 0.8, 1],
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
            apply_quantization=apply_quantization,
            levels=levels,
        )
        test_loss, test_acc = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device,
            apply_quantization=apply_quantization,
            levels=levels,
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
        if epoch == (epochs - 1):
            for name, param in model.named_parameters():
                if "weight" in name:
                    scale = calculate_scale(param.data)  # Calculate scale
                    param.data = custom_quantize(param.data, levels, scale)  # Apply ternary quantization
    return results
    # return model
