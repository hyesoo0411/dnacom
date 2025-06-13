#!/bin/bash
#SBATCH -J QAT_h$1_n$2
#SBATCH -p big_suma_rtx3090
#SBATCH -q big_qos
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --time 0-01:00:00
#SBATCH --output=runs/slog/train_h$1_n$2.out
#SBATCH --error=runs/slog/train_h$1_n$2.err

eval "$(conda shell.bash hook)"
conda activate DNAalign

echo "Running with hidden_units=$1 and noise=$2"

python MNIST_Code/model_train.py config/train.yaml \
  --hidden_units=$1 \
  --noise=$2 \
  --loss_curve_dir=./losscurve_new \
  --model_dir=./models \
  --output_suffix="hu${1}_n${2}"
