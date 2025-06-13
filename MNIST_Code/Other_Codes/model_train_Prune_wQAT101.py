"""
model_train_Prune_wQAT101.py: Pruning => 가중치만 -1 0 1로 양자화 
"""

import os

import torch

from torchvision import transforms

import data_setup, engine_Prune_wQAT101, model_builder, utils

import matplotlib.pyplot as plt

from typing import Tuple, Dict, List

import matplotlib


import torch.nn as nn
import torch.nn.functional as F

matplotlib.use("Agg")

# Set random seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Setup hyperparameters
NUM_EPOCHS = 20
BATCH_SIZE = 32
HIDDEN_UNITS = 80  # 200(10), 400(20), 800(40), 1600(80)
LEARNING_RATE = 0.001  # 0.001, 0.0005, 0.0001, 0.00005
apply_ternary_quantization = True
bias = False
pruning = True

# Setup directories
train_dir = "/home/lab/DNAcomputing/DNAcomputing/Dataset/train"
test_dir = "/home/lab/DNAcomputing/DNAcomputing/Dataset/test"

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create transforms
data_transform = transforms.Compose([transforms.Resize((7, 7)), transforms.ToTensor()])

# Create DataLoaders with help from data_setup.py
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir, test_dir=test_dir, transform=data_transform, batch_size=BATCH_SIZE
)


# Create model with help from model_builder.py
model = model_builder.WSmodel(input_shape=49, hidden_units=HIDDEN_UNITS, output_shape=1).to(device)

# Set loss and optimizer
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Start training with help from engine.py
results = engine_Prune_wQAT101.train_qat(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    epochs=NUM_EPOCHS,
    device=device,
    apply_ternary_quantization=apply_ternary_quantization,
)

# Save the model with help from utils.py
utils.save_model(
    model=model,
    target_dir="/home/lab/DNAcomputing/DNAcomputing/models",
    model_name=f"WSmodel_{HIDDEN_UNITS}units_lr{LEARNING_RATE}_epoch{NUM_EPOCHS}_Pruning_wQAT-101{apply_ternary_quantization}_bias{bias}.pth",
)


# def remove_zero_weight_units(model: nn.Module):
#     """
#     Prunes units (neurons) in the previous layer based on the classifier's weight.
#     If the classifier's weight for a unit is zero, the corresponding unit in the previous
#     layer is removed.

#     Args:
#         model: The trained WSmodel from which to remove zero-weight units.

#     Returns:
#         pruned_model: A new model with zero-weight units removed from the previous layer
#         and adjusted classifier.
#     """
#     new_layer_stack = []

#     # 1. Classifier의 가중치에서 0인 입력 뉴런 찾기
#     for name, module in model.classifier.named_children():
#         if isinstance(module, nn.Linear):
#             classifier_weight = module.weight.data

#             # classifier의 입력에 해당하는 가중치가 0인 뉴런 찾기 (입력 뉴런 기준)
#             non_zero_connections = torch.sum(
#                 classifier_weight != 0, dim=0
#             ).bool()  # 0이 아닌 연결 찾기 (입력 유닛 기준)
#             print(
#                 f"Classifier {name} - Removed {classifier_weight.size(1) - non_zero_connections.sum().item()} input connections"
#             )

#     # 2. Layer stack에서 classifier의 가중치가 0인 유닛 제거 (출력 뉴런 기준 Pruning)
#     for name, module in model.layer_stack.named_children():
#         if isinstance(module, nn.Linear):
#             weight = module.weight.data  # torch.Size([output_units, input_units])

#             # non_zero_connections에 해당하는 출력 뉴런만 남기기
#             pruned_weight = weight[non_zero_connections, :]  # 출력 뉴런 기준으로 Pruning
#             pruned_layer = nn.Linear(pruned_weight.size(1), pruned_weight.size(0), bias=False)
#             pruned_layer.weight.data = pruned_weight
#             new_layer_stack.append(pruned_layer)

#             print(f"Pruned layer_stack: {name} - Removed output units based on classifier weights")
#         else:
#             new_layer_stack.append(module)  # 다른 레이어는 그대로 유지

#     # 3. Classifier의 입력 유닛에서 Pruning 진행 (Layer stack의 출력 뉴런 기준)
#     new_classifier = []
#     for name, module in model.classifier.named_children():
#         if isinstance(module, nn.Linear):
#             classifier_weight = module.weight.data

#             # non_zero_connections에 해당하는 입력 유닛만 남기기
#             pruned_weight = classifier_weight[:, non_zero_connections]  # 입력 유닛 기준으로 Pruning
#             pruned_layer = nn.Linear(pruned_weight.size(1), pruned_weight.size(0), bias=False)
#             pruned_layer.weight.data = pruned_weight
#             new_classifier.append(pruned_layer)

#             print(
#                 f"Pruned classifier: {name} - Removed {classifier_weight.size(1) - pruned_weight.size(1)} input connections"
#             )
#         else:
#             new_classifier.append(module)  # 다른 레이어는 그대로 유지

#     # Pruned 모델 재구성
#     pruned_model = model_builder.WSmodel(
#         input_shape=model.layer_stack[1].in_features,  # 기존 모델의 input_shape 사용
#         hidden_units=new_layer_stack[1].out_features,  # Pruned 후 hidden_units 사용
#         output_shape=model.classifier[0].out_features,  # Pruned된 classifier의 출력 크기 사용
#     )

#     pruned_model.layer_stack = nn.Sequential(*new_layer_stack)
#     pruned_model.classifier = nn.Sequential(*new_classifier)

#     # Pruned된 모델의 레이어별 torch size 출력
#     print("\n--- Pruned Model Layer Sizes ---")
#     for name, module in pruned_model.named_modules():
#         if isinstance(module, nn.Linear):
#             print(f"Layer: {name}, Weight size: {module.weight.size()}")

#     return pruned_model


# pruned_model = remove_zero_weight_units(model)

# # Pruning 후 모델 저장
# # Save the model with help from utils.py
# utils.save_model(
#     model=pruned_model,
#     target_dir="/home/lab/DNAcomputing/DNAcomputing/models",
#     model_name=f"WSmodel_{HIDDEN_UNITS}units_lr{LEARNING_RATE}_epoch{NUM_EPOCHS}_wQAT-101{apply_ternary_quantization}_bias{bias}_Pruning{pruning}.pth",
# )


def plot_loss_curves(results: Dict[str, List[float]]):
    """Plots training curves of a results dictionary."""
    # Get the loss values of the results dictionary(training and test)
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    # Get the accuracy values of the results dictionary (training and test)
    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    # Figure out how many epochs there were
    epochs = range(len(results["train_loss"]))

    # Setup a plot
    plt.figure(figsize=(15, 7))

    # Plot the loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title(f"{HIDDEN_UNITS}_units, lr{LEARNING_RATE}, wQAT-101{apply_ternary_quantization}, bias{bias}:Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot the accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title(f"{HIDDEN_UNITS}_units, lr{LEARNING_RATE}, wQAT-101{apply_ternary_quantization}, bias{bias}:Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.savefig(
        f"/home/lab/DNAcomputing/DNAcomputing/LossCurve/LossCurve_{HIDDEN_UNITS}units_lr{LEARNING_RATE}_epoch{NUM_EPOCHS}_Pruning_wQAT-101{apply_ternary_quantization}_bias{bias}.png"
    )


plot_loss_curves(results)
