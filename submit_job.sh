#!/usr/bin/env bash
#SBATCH --partition=cpu_jobs                  # Ask for unkillable job
#SBATCH --cpus-per-task=4                     # Ask for 2 CPUs
#SBATCH --gres=gpu:0                          # Ask for 1 GPU
#SBATCH --mem=20G                             # Ask for 10 GB of RAM
#SBATCH --time=5:00:00                        # The job will run for 3 hours
#SBATCH -o /network/tmp1/<user>/slurm-%j.out  # Write the log on tmp1

# 1. Load your environment
source /network/home/rajkuman/.bashrc
module purge
module load anadaconda/3
source $CONDA_ACTIVATE
#conda env create -f environment.yml --prefix env
#conda env create -f causal-capacity-measures/environment.yml
conda activate ccm

# 2. Copy your dataset on the compute node
mkdir -p $SLURM_TMPDIR/data/MNIST/
mkdir $SLURM_TMPDIR/logs
mkdir $SLURM_TMPDIR/checkpoints
mkdir $SLURM_TMPDIR/results
cp -r /network/data1/mnist/processed $SLURM_TMPDIR/data/MNIST/

# 3. Launch your job
python run_experiment.py --root_dir=$SLURM_TMPDIR --use_cuda=True > $SLURM_TMPDIR/experiment.out

# 4. Copy whatever you want to save on $SCRATCH
cp $SLURM_TMPDIR/logs /network/tmp1/rajkuman/logs
cp $SLURM_TMPDIR/experiment.out /network/tmp1/rakjuman/experiment.out
#cp $SLURM_TMPDIR/checkpoints /network/tmp1/rajkuman/checkpoints