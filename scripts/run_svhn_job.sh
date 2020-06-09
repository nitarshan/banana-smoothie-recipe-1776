#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --time=3:00:00
#SBATCH --job-name=rgm

# 1. Create your environment locally
echo "Preparing environment"
module load python/3.8
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install -r requirements.txt

# 2. Copy your dataset on the compute node
# IMPORTANT: Your dataset must be compressed in one single file (zip, hdf5, ...)!!!
#wget http://ufldl.stanford.edu/housenumbers/train_32x32.mat
#wget http://ufldl.stanford.edu/housenumbers/test_32x32.mat
echo "Preparing data"
mkdir -p $SLURM_TMPDIR/data/
cp $SCRATCH/datasets/train_32x32.mat $SLURM_TMPDIR/data/
cp $SCRATCH/datasets/test_32x32.mat $SLURM_TMPDIR/data/

# 3. Launch your job
echo "Launching run"
seeds="0 1 2 3 4"
for seed in $seeds; do
  python train.py --seed=$seed --root_dir=$SLURM_TMPDIR --data_dir=$SLURM_TMPDIR/data --id=1 --dataset_type=SVHN --train_dataset_size=50000 --test_dataset_size=10000 "$@"
done

# 4. Move experiment outputs
mkdir -p $SCRATCH/causal-capacity-measures/results/
cp -r $SLURM_TMPDIR/results/* $SCRATCH/causal-capacity-measures/results/
