#!/usr/bin/env bash
#SBATCH --partition=unkillable
#SBATCH --job-name=complexity_lambda_search
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:titanrtx:1
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
optimizer='SGD_MOMENTUM'
measures='L2 PROD_OF_FRO SUM_OF_FRO PARAM_NORM PATH_NORM'
lambdas='0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0'
runs=3
global_idx=0
for measure in $measures; do
  for lambda in $lambdas; do
    if [ ! -d "/network/tmp1/rajkuman/logs/$model/$dataset/$optimizer/$measure/$lambda" ]; then
      for ((i=1;i<=runs;i++)); do
        let "global_idx++"
        python run_experiment.py single \
        --root_dir=$SLURM_TMPDIR \
        --model_type=$model \
        --dataset_type=$dataset \
        --optimizer_type=$optimizer \
        --lr=0.01 \
        --epochs=200 \
        --batch_size=100 \
        --complexity_type=$measure \
        --complexity_lambda=$lambda \
        --complexity_normalization \
        --use_cuda \
        --log_tensorboard &
      done
      wait
      rsync -r $SLURM_TMPDIR/logs/ /network/tmp1/rajkuman/logs
      rsync -r $SLURM_TMPDIR/results/ /network/tmp1/rajkuman/results
    fi
  done
done

# 4. To do on your local machine (not login)
# rsync -a --ignore-existing mila:/network/tmp1/rajkuman/logs/ ./logs
# rsync -a --ignore-existing mila:/network/tmp1/rajkuman/results/ ./results