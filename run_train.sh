#!/bin/bash

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate DNAalign

# Parameter combinations
hidden_units_list=(1600) #(200 400 800 1600)
noise_list=(0.1 0.5 0.8)

# Loop over all combinations
for hu in "${hidden_units_list[@]}"; do
  for noise in "${noise_list[@]}"; do
    job_name="qat_h${hu}_n${noise}"
    log_file="runs/slog/${job_name}_%j.out"
    error_file="runs/slog/${job_name}_%j.err"

    # Submit job to SLURM
    sbatch -p big_suma_rtx3090 -J "$job_name" -o "$log_file" -e "$error_file" \
      --gres=gpu:1 --nodes=1 --ntasks-per-node=64 --time 0-23:00:00 --qos=big_qos \
      --wrap="python MNIST_Code/model_train.py config/train.yaml \
        --hidden_units=${hu} \
        --noise=${noise} \
        --output_suffix=hu${hu}_n${noise}"
    sleep 1
  done
done
