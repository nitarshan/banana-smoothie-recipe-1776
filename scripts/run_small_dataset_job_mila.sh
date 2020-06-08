#!/usr/bin/env bash
#SBATCH --partition=long
#SBATCH --job-name=rgm
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:16gb:1
#SBATCH --mem=12G
#SBATCH --time=1:00:00

# 1. Load your environment
echo 'Load Environment'
source ~/.bashrc
module purge
module load anaconda/3
source $CONDA_ACTIVATE
conda activate base
conda activate rgm

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
seeds="0 1 2 3 4"
for seed in $seeds; do
  python run.py --seed=$seed --root_dir=$SLURM_TMPDIR --data_dir=$SLURM_TMPDIR/data --id=1 "$@"
done

# 4. Move experiment outputs
mkdir -p /network/tmp1/$USER/checkpoints/
mkdir -p /network/tmp1/$USER/results/
cp -r $SLURM_TMPDIR/checkpoints/* /network/tmp1/$USER/checkpoints/
cp -r $SLURM_TMPDIR/results/* /network/tmp1/$USER/results/
