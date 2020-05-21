#!/usr/bin/env bash

train_dataset_sizes="12500"
lrs="0.01 0.00631 0.003162 0.001585 0.001"
widths="8 10 12 14 16"
depths="2 3 4 5 6"
seeds="0 1 2 3 4"

for train_dataset_size in $train_dataset_sizes; do
  for lr in $lrs; do
    for width in $widths; do
      for depth in $depths; do
        for seed in $seeds; do
          sbatch ./scripts/run_mila.sh --seed=$seed --train_dataset_size=$train_dataset_size --lr=$lr --model_width=$width --model_depth=$depth
        done
      done
    done
  done
done
