import torch
import torch.nn as nn
from typing import Tuple
import data_setup, model_builder
from torchvision import transforms
from noise import apply_noise

# Set random seeds for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)

BATCH_SIZE = 32
# Setup directories
train_dir = "/home/hyesoo/DNAcomputing/Dataset/train"
test_dir = "/home/hyesoo/DNAcomputing/Dataset/test"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Transformations for the test set
data_transform = transforms.Compose([transforms.Resize((7, 7)), transforms.ToTensor()])

# Create DataLoaders with help from data_setup.py
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir, test_dir=test_dir, transform=data_transform, batch_size=BATCH_SIZE
)

def get_model_dimensions_from_state_dict(model_path: str) -> Tuple[int, int]:
    """
    Extracts input_shape and hidden_units from the state_dict of a saved model.
    """
    state_dict = torch.load(model_path)
    layer_stack_weight = state_dict["layer_stack.1.weight"]
    input_shape = layer_stack_weight.size(1)
    hidden_units = layer_stack_weight.size(0)
    return input_shape, hidden_units

def test_step(
    model: nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: nn.Module, device: torch.device
) -> Tuple[float, float]:
    """
    Evaluates a model on a test dataset.
    """
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y = y.float().unsqueeze(1)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()
            y_pred_class = (y_pred >= 0.5).long()
            test_acc += (y_pred_class == y.long()).sum().item() / len(y_pred_class)
    return test_loss / len(dataloader), test_acc / len(dataloader)

def check_pruning(model: nn.Module):
    """
    Checks the sizes of pruned layers in a model.
    """
    print("\n--- Pruned Model Layer Sizes ---")
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            print(f"Layer: {name}, Weight size: {module.weight.size()}")

def main():
    model_path = "/home/hyesoo/DNAcomputing/models/WSmodel_200units_lr0.001_epoch20_wQAT-101True_round_biasFalse_PruningTrue_scalefactor1.pth"
    input_shape, hidden_units = get_model_dimensions_from_state_dict(model_path)
    pruned_model = model_builder.WSmodel(input_shape=input_shape, hidden_units=hidden_units, output_shape=1)
    pruned_model.load_state_dict(torch.load(model_path))
    pruned_model = pruned_model.to(device)
    
    loss_fn = nn.BCEWithLogitsLoss()
    test_loss, test_acc = test_step(pruned_model, test_dataloader, loss_fn, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    check_pruning(pruned_model)

if __name__ == "__main__":
    main()
