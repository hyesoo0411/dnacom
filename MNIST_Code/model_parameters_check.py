import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib
import seaborn as sns

matplotlib.use("Agg")

# 텐서 값이 생략 없이 출력되도록 설정
torch.set_printoptions(edgeitems=torch.inf)

# pth 파일로 저장된 모델 로드 (모델의 state_dict 로드)
model_dir = "/home/ing07132/DNAcomputing/models"
modelname = "WSmodel_200units_lr0.001_epoch20_wQAT-101True_round_biasFalse_PruningTrue_scalefactor1_noise0.1"
modelfile = (
    "/home/ing07132/DNAcomputing/models/WSmodel_200units_lr0.001_epoch20_wQAT-101True_round_biasFalse_PruningTrue_scalefactor1_noise0.1.pth"
)
state_dict = torch.load(modelfile)


# state_dict를 텍스트 파일로 저장하는 함수
def save_state_dict_to_file(state_dict, filename="state_dict.txt"):
    with open(filename, "w") as f:
        for layer_name, param in state_dict.items():
            # 레이어 이름 출력
            f.write(f"Key: {layer_name}\n")
            # 가중치 값 출력
            f.write(f"Value:\n{param}\n")
            # print(param.size())
            # 레이어 구분을 위한 빈 줄
            f.write("\n\n")
    print(f"State dict saved to {filename}")


# state_dict를 텍스트 파일로 저장
save_state_dict_to_file(state_dict, f"/home/ing07132/DNAcomputing/State_dict/{modelname}.txt")


# 1. 모델 가중치 로드
state_dict = torch.load(modelfile, map_location=torch.device("cpu"))  # map_location을 통해 CPU에서 로드

# 2. 가중치 저장할 디렉토리 생성
save_dir = "/home/ing07132/DNAcomputing/weight_distribution_plots"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 3. 레이어별로 가중치 추출 및 분포 그래프 그리기
for layer_name, param in state_dict.items():
    if "weight" in layer_name:  # 가중치만 확인
        weights = param.data.numpy()  # 텐서를 numpy 배열로 변환

        # Flatten the weights array if it's multidimensional (for histogram)
        weights = weights.flatten()

        # 4. 히스토그램을 그려서 저장
        plt.figure(figsize=(8, 6))
        plt.hist(weights, bins=30, alpha=0.75, edgecolor="black")
        plt.title(f"Weight Distribution - {layer_name}")
        plt.xlabel("Weight")
        plt.ylabel("Frequency")
        plt.grid(True)

        # Save plot for each layer
        plot_filename = os.path.join(save_dir, f"{modelname}_weight_distribution_{layer_name}.png")
        plt.savefig(plot_filename)

print(f"Weight distribution plots saved in directory: {save_dir}")

# 추가: layer_stack.1.weight를 이용해 히트맵 그리기
save_dir = "/home/ing07132/DNAcomputing/Heatmap"
if "layer_stack.1.weight" in state_dict:
    layer_weights = state_dict["layer_stack.1.weight"].cpu().numpy()

    # 히트맵의 범위 설정을 위해 전체 가중치에서 최솟값과 최댓값 계산
    vmin = layer_weights.min()
    vmax = layer_weights.max()

    # 6개의 hidden nodes 각각에 대해 7x7 히트맵을 하나의 figure에 그리기
    plt.figure(figsize=(15, 10))
    for i in range(4):
        weight_matrix = layer_weights[i].reshape(7, 7)
        plt.subplot(2, 3, i + 1)  # 2행 3열 형태로 배치
        sns.heatmap(weight_matrix, annot=False, cmap='coolwarm', cbar=True, vmin=vmin, vmax=vmax)
        plt.title(f'Hidden Node {i+1} Weights')
    
    plt.suptitle('200 => 4 units', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    combined_heatmap_filename = os.path.join(save_dir, f"{modelname}_combined_hidden_nodes_heatmap.png")
    plt.savefig(combined_heatmap_filename)
    plt.close()


print(f"Heatmaps for hidden nodes saved in directory: {save_dir}")


# def count_pruned_units_per_layer(model_path):
#     # 모델의 state_dict 불러오기
#     state_dict = torch.load(model_path, weights_only=True)

#     # 각 레이어의 unit(뉴런) 개수 세기
#     unit_counts = {}
#     for layer_name, weights in state_dict.items():
#         if "weight_orig" in layer_name:  # 가중치가 있는 레이어만 처리
#             # 가중치 텐서의 shape에서 첫 번째 차원 크기는 전체 유닛 수
#             total_units = weights.shape[0]

#             # 0이 아닌 가중치가 있는 유닛의 개수 계산
#             non_zero_units = (weights.abs().sum(dim=1) != 0).sum().item()

#             unit_counts[layer_name] = {
#                 'total_units': total_units,
#                 'non_zero_units': non_zero_units
#             }

#     return unit_counts


# unit_counts = count_pruned_units_per_layer(modelfile)

# # 각 레이어별 전체 유닛 개수와 Pruning 후 남은 유닛 개수 출력
# for layer, counts in unit_counts.items():
#     print(f"Layer: {layer}, Total Units: {counts['total_units']}, Non-zero Units after Pruning: {counts['non_zero_units']}")
