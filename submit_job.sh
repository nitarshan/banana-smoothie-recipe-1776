#!/usr/bin/env bash
#SBATCH --partition=main-grace
#SBATCH --job-name=test
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:titanxp:1
#SBATCH --mem=20G
#SBATCH --time=4:00:00
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
mkdir -p $SLURM_TMPDIR/data/MNIST/
mkdir $SLURM_TMPDIR/logs
mkdir $SLURM_TMPDIR/checkpoints
mkdir $SLURM_TMPDIR/results
cp -r /network/data1/mnist/processed $SLURM_TMPDIR/data/MNIST/
cp -r /network/data1/cifar/cifar-10-batches-py $SLURM_TMPDIR/data/

# 3. Launch your job
models='DEEP'
datasets='MNIST'
optimizers='ADAM SGD'
measures='L2 PROD_OF_FRO SUM_OF_FRO PARAM_NORM PATH_NORM'
for model in $models
do
  for dataset in $datasets
  do
    for optimizer in $optimizers
    do
      for measure in $measures
      do
        echo "/network/tmp1/rajkuman/logs/$model/$dataset/$optimizer/$measure"
      done
      mv -v $SLURM_TMPDIR/logs/* /network/tmp1/rajkuman/logs
    done
  done
done

if [ ! -d "/network/tmp1/rajkuman/logs/DEEP/MNIST/ADAM/" ] 
then
echo "[SLURM SCRIPT] DEEP MNIST ADAM"
python run_experiment.py single $SLURM_TMPDIR DEEP MNIST ADAM 0.001 50 256 L2 0 --log_tensorboard &
python run_experiment.py single $SLURM_TMPDIR DEEP MNIST ADAM 0.001 50 256 PROD_OF_FRO 0 --log_tensorboard &
python run_experiment.py single $SLURM_TMPDIR DEEP MNIST ADAM 0.001 50 256 SUM_OF_FRO 0 --log_tensorboard &
wait
python run_experiment.py single $SLURM_TMPDIR DEEP MNIST ADAM 0.001 200 256 PARAM_NORM 0 --log_tensorboard &
python run_experiment.py single $SLURM_TMPDIR DEEP MNIST ADAM 0.001 200 256 PATH_NORM 0 --log_tensorboard &
wait
echo "[SLURM SCRIPT] DEEP MNIST ADAM completed"
mv -v $SLURM_TMPDIR/logs/* /network/tmp1/rajkuman/logs
fi

if [ ! -d "/network/tmp1/rajkuman/logs/DEEP/MNIST/SGD/" ] 
then
echo "[SLURM SCRIPT] DEEP MNIST SGD"
python run_experiment.py single $SLURM_TMPDIR DEEP MNIST SGD 0.001 200 256 L2 0 --log_tensorboard &
python run_experiment.py single $SLURM_TMPDIR DEEP MNIST SGD 0.001 200 256 PROD_OF_FRO 0 --log_tensorboard &
python run_experiment.py single $SLURM_TMPDIR DEEP MNIST SGD 0.001 200 256 SUM_OF_FRO 0 --log_tensorboard &
wait
python run_experiment.py single $SLURM_TMPDIR DEEP MNIST SGD 0.001 200 256 PARAM_NORM 0 --log_tensorboard &
python run_experiment.py single $SLURM_TMPDIR DEEP MNIST SGD 0.001 200 256 PATH_NORM 0 --log_tensorboard &
wait
echo "[SLURM SCRIPT] DEEP MNIST SGD completed"
mv -v $SLURM_TMPDIR/logs/* /network/tmp1/rajkuman/logs
fi

if [ ! -d "/network/tmp1/rajkuman/logs/DEEP/CIFAR10/ADAM/" ] 
then
echo "[SLURM SCRIPT] DEEP CIFAR10 SGD"
python run_experiment.py single $SLURM_TMPDIR DEEP CIFAR10 ADAM 0.001 200 256 L2 0 --log_tensorboard &
python run_experiment.py single $SLURM_TMPDIR DEEP CIFAR10 ADAM 0.001 200 256 PROD_OF_FRO 0 --log_tensorboard &
python run_experiment.py single $SLURM_TMPDIR DEEP CIFAR10 ADAM 0.001 200 256 SUM_OF_FRO 0 --log_tensorboard &
wait
python run_experiment.py single $SLURM_TMPDIR DEEP CIFAR10 ADAM 0.001 200 256 PARAM_NORM 0 --log_tensorboard &
python run_experiment.py single $SLURM_TMPDIR DEEP CIFAR10 ADAM 0.001 200 256 PATH_NORM 0 --log_tensorboard &
wait
echo "[SLURM SCRIPT] DEEP CIFAR10 SGD completed"
mv -v $SLURM_TMPDIR/logs/* /network/tmp1/rajkuman/logs
fi

if [ ! -d "/network/tmp1/rajkuman/logs/DEEP/CIFAR10/SGD/" ] 
then
echo "[SLURM SCRIPT] DEEP CIFAR10 ADAM"
python run_experiment.py single $SLURM_TMPDIR DEEP CIFAR10 SGD 0.001 200 256 L2 0 --log_tensorboard &
python run_experiment.py single $SLURM_TMPDIR DEEP CIFAR10 SGD 0.001 200 256 PROD_OF_FRO 0 --log_tensorboard &
python run_experiment.py single $SLURM_TMPDIR DEEP CIFAR10 SGD 0.001 200 256 SUM_OF_FRO 0 --log_tensorboard &
wait
python run_experiment.py single $SLURM_TMPDIR DEEP CIFAR10 SGD 0.001 200 256 PARAM_NORM 0 --log_tensorboard &
python run_experiment.py single $SLURM_TMPDIR DEEP CIFAR10 SGD 0.001 200 256 PATH_NORM 0 --log_tensorboard &
wait
echo "[SLURM SCRIPT] DEEP CIFAR10 ADAM completed"
mv -v $SLURM_TMPDIR/logs/* /network/tmp1/rajkuman/logs
fi

# On local machine (not login)
# rsync -a --ignore-existing mila:/network/tmp1/rajkuman/logs/ ./logs