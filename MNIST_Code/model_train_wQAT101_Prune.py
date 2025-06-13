"""
model_train_wQAT101_Prune.py: 가중치만 -1 0 1로 양자화 => Pruning
"""

import os
from omegaconf import OmegaConf
import torch
from torchvision import transforms
import data_setup, engine_wQAT101_Prune, model_builder, utils
import matplotlib.pyplot as plt
from typing import Dict, List
import torch.nn as nn
import sys

# Load configurations
if len(sys.argv) < 2:
    raise ValueError("Please provide the path to the configuration file as the first argument.")
config_path = sys.argv[1]
config = OmegaConf.load(config_path)

# Set random seeds
torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)

# Setup target device
device = config.device if torch.cuda.is_available() else "cpu"

# Create transforms
data_transform = transforms.Compose([
    transforms.Resize((7, 7)),
    transforms.ToTensor()
])

# Setup directories
train_dir = config.directories.train_dir
test_dir = config.directories.test_dir

# Create DataLoaders
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=data_transform,
    batch_size=config.hyperparameters.batch_size
)

# Create model
model = model_builder.WSmodel(
    input_shape=config.model.input_shape,
    hidden_units=config.hyperparameters.hidden_units,
    output_shape=config.model.output_shape,
    noise=config.hyperparameters.noise
).to(device)

# Set loss and optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.hyperparameters.learning_rate)

# Train the model
results = engine_wQAT101_Prune.train_qat(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    epochs=config.hyperparameters.num_epochs,
    device=device,
    apply_ternary_quantization=config.hyperparameters.apply_ternary_quantization,
)

# Save the model
utils.save_model(
    model=model,
    target_dir=config.directories.model_dir,
    model_name=f"WSmodel_{config.hyperparameters.hidden_units}units_lr{config.hyperparameters.learning_rate}_epoch{config.hyperparameters.num_epochs}_wQAT-101{config.hyperparameters.apply_ternary_quantization}_round_bias{config.hyperparameters.bias}_scalefactor{config.hyperparameters.scale_factor}_noise{config.hyperparameters.noise}.pth",
)

def remove_zero_weight_units(model: nn.Module):
    """
    Prunes units (neurons) in the previous layer based on the classifier's weight.
    If the classifier's weight for a unit is zero, the corresponding unit in the previous
    layer is removed.

    Args:
        model: The trained WSmodel from which to remove zero-weight units.

    Returns:
        pruned_model: A new model with zero-weight units removed from the previous layer
        and adjusted classifier.
    """
    new_layer_stack = []

    # 1. Classifier의 입력 연결 중 0이 아닌 연결을 찾기
    for name, module in model.classifier.named_children():
        if isinstance(module, nn.Linear):
            classifier_weight = module.weight.data  # Classifier의 가중치 텐서

            # 가중치가 0이 아닌 입력 뉴런의 마스크 생성 (열 기준)
            non_zero_connections = torch.sum(classifier_weight != 0, dim=0).bool()  
            # 마스크는 True/False로 이루어져 있어, True는 0이 아닌 연결을 의미
            print(
                f"Classifier {name} - Removed {classifier_weight.size(1) - non_zero_connections.sum().item()} input connections"
            )

    # 2. Hidden layer에서 classifier와 연결된 가중치가 0인 유닛 제거
    # (classifier와 연결된 hidden layer의 출력 뉴런 중 0이 아닌 연결을 유지)
    for name, module in model.layer_stack.named_children():
        if isinstance(module, nn.Linear):
            weight = module.weight.data  # Hidden layer의 가중치 텐서 (output_units, input_units)

            # non_zero_connections에 해당하는 출력 뉴런만 남김
            pruned_weight = weight[non_zero_connections, :]  
            pruned_layer = nn.Linear(pruned_weight.size(1), pruned_weight.size(0), bias=False)
            pruned_layer.weight.data = pruned_weight  # Pruned된 가중치를 새로운 레이어에 할당
            new_layer_stack.append(pruned_layer)

            print(f"Pruned layer_stack: {name} - Removed output units based on classifier weights")
        else:
            new_layer_stack.append(module)  # Linear가 아닌 다른 레이어는 그대로 추가

    # 3. 앞서 Pruning된 hidden layer의 출력 유닛에 맞춰 classifier의 입력 유닛을 Pruning
    # (hidden layer의 output과 classifier의 input 크기를 맞추기 위해)
    new_classifier = []
    for name, module in model.classifier.named_children():
        if isinstance(module, nn.Linear):
            classifier_weight = module.weight.data  # Classifier의 가중치 텐서 (output_units, input_units)

            # non_zero_connections에 해당하는 입력 유닛만 남깁니다
            pruned_weight = classifier_weight[:, non_zero_connections]  
            # Pruning된 hidden layer의 출력에 맞춰 classifier의 입력을 조정
            pruned_layer = nn.Linear(pruned_weight.size(1), pruned_weight.size(0), bias=False)
            pruned_layer.weight.data = pruned_weight  # Pruned된 가중치를 새로운 레이어에 할당
            new_classifier.append(pruned_layer)

            print(
                f"Pruned classifier: {name} - Removed {classifier_weight.size(1) - pruned_weight.size(1)} input connections"
            )
        else:
            new_classifier.append(module)  # Linear가 아닌 다른 레이어는 그대로 추가

    # Pruned 모델 재구성
    pruned_model = model_builder.WSmodel(
        input_shape=model.layer_stack[1].in_features,  # 기존 모델의 input_shape 사용
        hidden_units=new_layer_stack[1].out_features,  # Pruned 후 hidden_units 사용
        output_shape=model.classifier[0].out_features,  # Pruned된 classifier의 출력 크기 사용
        noise=config.hyperparameters.noise
    )

    pruned_model.layer_stack = nn.Sequential(*new_layer_stack)
    pruned_model.classifier = nn.Sequential(*new_classifier)

    # Pruned된 모델의 레이어별 torch size 출력
    print("\n--- Pruned Model Layer Sizes ---")
    for name, module in pruned_model.named_modules():
        if isinstance(module, nn.Linear):
            print(f"Layer: {name}, Weight size: {module.weight.size()}")

    return pruned_model


pruned_model = remove_zero_weight_units(model)

# Save the pruned model
utils.save_model(
    model=pruned_model,
    target_dir=config.directories.model_dir,
    model_name=f"WSmodel_{config.hyperparameters.hidden_units}units_lr{config.hyperparameters.learning_rate}_epoch{config.hyperparameters.num_epochs}_wQAT-101{config.hyperparameters.apply_ternary_quantization}_round_bias{config.hyperparameters.bias}_Pruning{config.hyperparameters.pruning}_scalefactor{config.hyperparameters.scale_factor}_noise{config.hyperparameters.noise}.pth",
)

# Plot loss curves
def plot_loss_curves(results: Dict[str, List[float]]):
    loss = results["train_loss"]
    test_loss = results["test_loss"]
    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]
    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    # Plot the loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title(f"Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot the accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title(f"Accuracy")
    plt.xlabel("Epochs")
    plt.legend()

    plt.savefig(
        f"{config.directories.loss_curve_dir}/LossCurve_{config.hyperparameters.hidden_units}units_lr{config.hyperparameters.learning_rate}_epoch{config.hyperparameters.num_epochs}_wQAT-101{config.hyperparameters.apply_ternary_quantization}_round_bias{config.hyperparameters.bias}_scalefactor{config.hyperparameters.scale_factor}_noise{config.hyperparameters.noise}.png"
    )

plot_loss_curves(results)
