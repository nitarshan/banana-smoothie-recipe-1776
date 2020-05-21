#!/usr/bin/env bash
#SBATCH --partition=long
#SBATCH --job-name=ccm
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=12G
#SBATCH --time=2:00:00

# 1. Load your environment
echo 'Load Environment'
source ~/.bashrc
module purge
module load anaconda/3
source $CONDA_ACTIVATE
conda activate base
conda activate ccm

# 2. Prepare directories and copy dataset onto the compute node
echo "Loading Datasets!"
mkdir -p $SLURM_TMPDIR/data/
# MNIST
cp -r /network/datasets/torchvision/MNIST $SLURM_TMPDIR/data/
# CIFAR-10
cp /network/datasets/cifar10/cifar-10-python.tar.gz $SLURM_TMPDIR/data/
tar -xvzf $SLURM_TMPDIR/data/cifar-10-python.tar.gz -C $SLURM_TMPDIR/data/
# SVHN
# cp -r /network/data1/svhn $SLURM_TMPDIR/data/

# 3. Launch your job
echo "Launching Experiment"
python run.py --data_dir=$SLURM_TMPDIR/data --id=1 "$@"
