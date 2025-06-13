import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
import data_setup, engine, model_builder, utils
import pickle
import torch
import model_builder  # model_builder.py에서 모델 클래스가 정의되어 있다고 가정


# 1. 저장된 모델 불러오기
def load_model(model_path: str, input_shape: int, hidden_units: int, output_shape: int):
    # model_builder.py에 TinyVGG 모델이 있다고 가정
    model = model_builder.TinyVGG(input_shape=input_shape, hidden_units=hidden_units, output_shape=output_shape)

    # 저장된 모델 가중치 로드
    model.load_state_dict(torch.load(model_path))

    # 모델을 평가 모드로 설정 (필수)
    model.eval()
    return model


# 2. 저장된 결과 불러오기
def load_results(results_path: str) -> Dict[str, List[float]]:
    with open(results_path, "rb") as f:
        results = pickle.load(f)
    return results


# 저장된 결과 파일 경로 및 모델 파일 경로 지정
model_path = "/home/lab/DNAcomputing/models/WSmode100.pth"  # 저장된 모델의 경로

# 모델 및 결과 불러오기
input_shape = 49  # 예: 7x7 이미지
hidden_units = 100
output_shape = 2  # class_num

loaded_model = load_model(model_path, input_shape, hidden_units, output_shape)


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
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot the accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()


NUM_EPOCHS = 5
BATCH_SIZE = 64
HIDDEN_UNITS = 100
LEARNING_RATE = 0.001
class_num = 2
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE_PATH = "/home/lab/DNAcomputing/models/WSmode100.pth"

loaded_model = model_builder.WSmodel100(input_shape=49, hidden_units=HIDDEN_UNITS, output_shape=class_num).to(device)

# Load in the save state_dicts()
loaded_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
# Send the model to the target device
loaded_model.to(device)

loaded_model_results = eval_model(
    model=loaded_model, data_loader=test_dataloader, loss_fn=loss_fn, accuracy_fn=accuracy_fn
)

loaded_model_2_results
