#!/usr/bin/env bash
#SBATCH --partition=unkillable
#SBATCH --job-name=verify_lagrangian
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:titanxp:1
#SBATCH --mem=20G
#SBATCH --time=24:00:00
#SBATCH --output /network/tmp1/rajkuman/slurm-%j.out
#SBATCH --error /network/tmp1/rajkuman/slurm-error-%j.out

# 1. Load your environment
source /network/home/rajkuman/.bashrc
module purge
module load anaconda/3
source $CONDA_ACTIVATE
conda activate ccm

# 2. Prepare directories and copy dataset onto the compute node
mkdir -p /network/tmp1/rajkuman/logs
mkdir -p /network/tmp1/rajkuman/results
mkdir -p $SLURM_TMPDIR/data/MNIST/
mkdir $SLURM_TMPDIR/logs
mkdir $SLURM_TMPDIR/checkpoints
mkdir $SLURM_TMPDIR/results
cp -r /network/data1/mnist/processed $SLURM_TMPDIR/data/MNIST/
cp -r /network/data1/cifar/cifar-10-batches-py $SLURM_TMPDIR/data/

# 3. Launch your job
model='DEEP'
dataset='MNIST'
optimizer='SGD SGD_MOMENTUM ADAM'
measures='L2 PROD_OF_FRO SUM_OF_FRO PARAM_NORM PATH_NORM'
runs=2
global_idx=0
count=-1
jobs_per_gpu=4
for measure in $measures; do
let "count++"
for mode in $model; do
for optim in $optimizer; do
  for ((i=1;i<=runs;i++)); do
    let "global_idx++"
    python run_experiment.py single \
    --root_dir=$SLURM_TMPDIR \
    --model_type=$mode \
    --dataset_type=$dataset \
    --optimizer_type=$optim \
    --lr=0.01 \
    --epochs=200 \
    --batch_size=100 \
    --complexity_type=$measure \
    --lagrangian_target=0 \
    --lagrangian_start_epoch=0 \
    --lagrangian_start_mu=1e-6 \
    --lagrangian_tolerance=0.01 \
    --lagrangian_patience_batches=250 \
    --lagrangian_improvement_rate=0.75 \
    --lagrangian_start_lambda=0 \
    --lagrangian_lambda_omega=1e-3 \
    --comet_api_key=$COMET_API_KEY \
    --comet_tag='default_values_mnist_4' \
    --use_cuda &
    if (( $global_idx % $jobs_per_gpu == 0 )); then
      wait
      rsync -r $SLURM_TMPDIR/results/ /network/tmp1/rajkuman/results
    fi
  done
done
done
done
wait
rsync -r $SLURM_TMPDIR/results/ /network/tmp1/rajkuman/results

# 4. To do on your local machine (not login)
# mkdir -p results/verify_lagrangian
# rsync -a --ignore-existing mila:/network/tmp1/rajkuman/results/ ./results/verify_lagrangian