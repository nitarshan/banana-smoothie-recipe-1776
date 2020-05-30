#!/usr/bin/env bash
if [ "$1" = "mila" ]
then
  rsync --exclude='.git/' --exclude='*__pycache__' -ruv ./ $1:~/causal-capacity-measures/
else
  rsync --exclude='.git/' --exclude='*__pycache__' -ruv ./ $1:~/scratch/causal-capacity-measures/
fi
