#!/usr/bin/env bash
rsync --exclude='.git/' -ruv ./ mila:~/causal-capacity-measures/
