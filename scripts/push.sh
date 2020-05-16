#!/usr/bin/env bash
if [ "$1" = "mila" ]
then
  rsync --exclude='.git/' -ruv ./ $1:~/causal-capacity-measures/
else
  rsync --exclude='.git/' -ruv ./ $1:~/scratch/causal-capacity-measures/
fi
