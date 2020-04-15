#!/usr/bin/env bash
#SBATCH --partition=unkillable
#SBATCH --job-name=verify_lagrangian
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:titanxp:1
#SBATCH --mem=20G
#SBATCH --time=24:00:00
#SBATCH --output /network/tmp1/$USER/slurm-%j.out
#SBATCH --error /network/tmp1/$USER/slurm-error-%j.out

# 1. Load your environment
source /network/home/$USER/.bashrc
module purge
module load anaconda/3
source $CONDA_ACTIVATE
conda activate ccm

# 2. Prepare directories and copy dataset onto the compute node
mkdir -p /network/tmp1/$USER/logs
mkdir -p /network/tmp1/$USER/results
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
measures='L2 LOG_PROD_OF_FRO LOG_SUM_OF_FRO PARAM_NORM PATH_NORM'
targets=(19.73 8096.64 26.50 381.56 45.56) # See complexity_lambda_analysis.ipynb
runs=3
global_idx=0
count=-1
jobs_per_gpu=4
for measure in $measures; do
  let "count++"
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
    --complexity_lambda=None \
    --lagrangian_type='AUGMENTED' \
    --lagrangian_target=${targets[count]} \
    --lagrangian_start_epoch=0 \
    --lagrangian_start_mu=1e-6 \
    --lagrangian_tolerance=1e-3 \
    --lagrangian_patience_batches=100 \
    --lagrangian_improvement_rate=0.75 \
    --lagrangian_start_lambda=0 \
    --lagrangian_convergence_tolerance=1e-3 \
    --comet_api_key=$COMET_API_KEY \
    --comet_tag='verify_lagrangian_augmented' \
    --use_cuda &
    if (( $global_idx % $jobs_per_gpu == 0 )); then
      wait
      rsync -r $SLURM_TMPDIR/results/ /network/tmp1/$USER/results
    fi
  done
done
wait
rsync -r $SLURM_TMPDIR/results/ /network/tmp1/$USER/results

# 4. To do on your local machine (not login)
# mkdir -p results/verify_lagrangian
# rsync -a --ignore-existing mila:/network/tmp1/$USER/results/ ./results/verify_lagrangian