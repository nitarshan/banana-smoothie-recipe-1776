#!/usr/bin/env bash

lrs="0.01 0.00631 0.003162 0.001585 0.001"
widths="8 10 12 14 16"
depths="2 3 4 5 6"

for lr in $lrs; do
for width in $widths; do
for depth in $depths; do
  sbatch ./scripts/run_svhn_job.sh --lr=$lr --model_width=$width --model_depth=$depth
done
done
done
