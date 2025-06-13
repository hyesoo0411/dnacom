"""
model_train_QAT-101.py: activation (이건 -1 0 1 양자화 아님 ), weight 둘다 양자화
QAT-101True: 3중 양자화 (-1, 0, 1)
QAT-101False: 2중 양자화 (-128에서 127) 
"""

import os

import torch

from torchvision import transforms

import data_setup, engine, model_builder, utils

import matplotlib.pyplot as plt

from typing import Tuple, Dict, List

# Set random seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Setup hyperparameters
NUM_EPOCHS = 20
BATCH_SIZE = 32
HIDDEN_UNITS = 800  # 100, 200, 400, 800, 1600
LEARNING_RATE = 0.001  # 0.001, 0.0005, 0.0001, 0.00005
apply_ternary_quantization = True
bias = False

# Setup directories
train_dir = "/home/lab/DNAcomputing/Dataset/train"
test_dir = "/home/lab/DNAcomputing/Dataset/test"

# Setup target device
cuda_device = torch.device("cuda:0")
cpu_device = torch.device("cpu:0")

# Create transforms
data_transform = transforms.Compose([transforms.Resize((7, 7)), transforms.ToTensor()])

# Create DataLoaders with help from data_setup.py
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir, test_dir=test_dir, transform=data_transform, batch_size=BATCH_SIZE
)


model_dir = "/home/lab/DNAcomputing/models"
model_filename = f"WSmodel_{HIDDEN_UNITS}units_lr0.001_epoch20_bias{bias}.pth"
quantized_model_filename = f"WSmodel_{HIDDEN_UNITS}units_lr{LEARNING_RATE}_epoch{NUM_EPOCHS}_QAT-101{apply_ternary_quantization}_bias{bias}.pth"
model_filepath = os.path.join(model_dir, model_filename)
quantized_model_filepath = os.path.join(model_dir, quantized_model_filename)

# Create model with help from model_builder.py
model = model_builder.WSmodel(input_shape=49, hidden_units=HIDDEN_UNITS, output_shape=len(class_names)).to(cuda_device)


# Load a pretrained model
def load_model(model, model_filepath, device):
    model.load_state_dict(torch.load(model_filepath, map_location=device))
    return model


model = load_model(model=model, model_filepath=model_filepath, device=cuda_device)
model.to(cpu_device)
model.train()
quantized_model = model_builder.WSmodelQAT(WSmodel=model)
quantization_config = torch.quantization.get_default_qconfig("fbgemm")
quantized_model.qconfig = quantization_config
print("configure quantization setting:")
print(quantized_model.qconfig)

torch.quantization.prepare_qat(quantized_model, inplace=True)

# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# Start training with help from engine.py
print("Training QAT Model...")
quantized_model.to(cuda_device)
results, quantized_model = engine.train_qat(
    model=quantized_model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    epochs=NUM_EPOCHS,
    device=cuda_device,
    apply_ternary_quantization=apply_ternary_quantization,
)

quantized_model.to(cpu_device)
quantized_model = torch.quantization.convert(quantized_model, inplace=True)
quantized_model.eval()

# Print quantized model.
print("Printing quantized_model")
print(quantized_model)

# Save the model with help from utils.py
utils.save_model(
    model=quantized_model,
    target_dir=model_dir,
    model_name=quantized_model_filename,
)


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
    plt.title(
        f"{HIDDEN_UNITS}_units, lr{LEARNING_RATE}, Epoch{NUM_EPOCHS}, QAT(-1,0,1){apply_ternary_quantization}, bias{bias}:Loss"
    )
    plt.xlabel("Epochs")
    plt.legend()

    # Plot the accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title(
        f"{HIDDEN_UNITS}_units, lr{LEARNING_RATE}, Epoch{NUM_EPOCHS}, QAT(-1,0,1){apply_ternary_quantization}, bias{bias}:Accuracy"
    )
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()
    plt.savefig(
        f"/home/lab/DNAcomputing/LossCurve/LossCurve_{HIDDEN_UNITS}units_lr{LEARNING_RATE}_epoch{NUM_EPOCHS}_QAT-101{apply_ternary_quantization}_bias{bias}.png"
    )


plot_loss_curves(results)
