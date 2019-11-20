#!/usr/bin/env bash
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:titanxp:1
#SBATCH --mem=20G
#SBATCH --time=5:00:00
#SBATCH --output /network/tmp1/rajkuman/slurm-%j.out

# 1. Load your environment
source /network/home/rajkuman/.bashrc
module purge
module load anadaconda/3
source $CONDA_ACTIVATE
#conda env create -f environment.yml --prefix env
#conda env create -f causal-capacity-measures/environment.yml
conda activate ccm

# 2. Prepare directories and copy dataset onto the compute node
mkdir -p $SLURM_TMPDIR/data/MNIST/
mkdir $SLURM_TMPDIR/logs
mkdir $SLURM_TMPDIR/checkpoints
mkdir $SLURM_TMPDIR/results
cp -r /network/data1/mnist/processed $SLURM_TMPDIR/data/MNIST/
cp -r /network/data1/cifar/cifar-10-batches-py $SLURM_TMPDIR/data/

# 3. Launch your job
touch $SLURM_TMPDIR/experiment.log
python run_experiment.py single $SLURM_TMPDIR CONV ADAM 0.001 CIFAR10 L2 0.1 200 128 50 --use_cuda --log_tensorboard >> $SLURM_TMPDIR/experiment.log &

# 4. Copy whatever you want to save on $SCRATCH
cp -r $SLURM_TMPDIR/logs/. /network/tmp1/rajkuman/logs
cp -r $SLURM_TMPDIR/checkpoints/. /network/tmp1/rajkuman/checkpoints
cp -r $SLURM_TMPDIR/results/. /network/tmp1/rajkuman/results
cp $SLURM_TMPDIR/experiment.out /network/tmp1/rakjuman/experiment.out
