#!/usr/bin/env bash

train_dataset_sizes="6250 12500"
lrs="0.01 0.00631 0.003162 0.001585 0.001"
widths="8 10 12 14 16"
depths="2 3 4 5 6"

for train_dataset_size in $train_dataset_sizes; do
  for lr in $lrs; do
    for width in $widths; do
      for depth in $depths; do
        sbatch ./scripts/run_small_dataset_job_mila.sh --train_dataset_size=$train_dataset_size --lr=$lr --model_width=$width --model_depth=$depth
      done
    done
  done
done
