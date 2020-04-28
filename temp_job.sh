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
mkdir -p /network/tmp1/$USER/results
mkdir -p $SLURM_TMPDIR/data/MNIST/
mkdir -p $SLURM_TMPDIR/results/
cp -r /network/data1/mnist/processed $SLURM_TMPDIR/data/MNIST/
cp -r /network/data1/cifar/cifar-10-batches-py $SLURM_TMPDIR/data/
# cp -r /network/data1/svhn $SLURM_TMPDIR/data/

# 3. Launch your job
echo "Launching Experiment"
tag="nin_test"
model="NIN"
dataset="CIFAR10"
optims="SGD_MOMENTUM"
lagrangian_types="NONE"
measures="L2"
targets="7"
ce_target="0.05"
lrs="0.01"
widths="2"
depths="2"
global_idx=0
jobs_per_gpu=5

for measure in $measures; do
for lagrangian_type in $lagrangian_types; do
for target in $targets; do
for optim in $optims; do
for lr in $lrs; do
for width in $widths; do
for depth in $depths; do
let "global_idx++"
python run_experiment.py single \
--root_dir=$SLURM_TMPDIR \
--model_type=$model \
--model_depth=$depth \
--model_width=$width \
--dataset_type=$dataset \
--optimizer_type=$optim \
--lr=$lr \
--epochs=150 \
--batch_size=128 \
--complexity_type=$measure \
--complexity_lambda=None \
--lagrangian_type=$lagrangian_type \
--lagrangian_target=$target \
--lagrangian_start_epoch=0 \
--lagrangian_start_mu=1e-2 \
--lagrangian_tolerance=0.1 \
--lagrangian_patience_batches=200 \
--lagrangian_improvement_rate=0.75 \
--lagrangian_start_lambda=0 \
--global_convergence_method=leq \
--lagrangian_convergence_tolerance=1e-4 \
--global_convergence_tolerance=1e-8 \
--global_convergence_patience=30 \
--global_convergence_target=$ce_target \
--global_convergence_evaluation_freq_milestones=[0.09,0.07,0.06] \
--comet_api_key=$COMET_API_KEY \
--log_epoch_freq=10 \
--comet_tag=$tag \
--use_cuda \
--use_wandb \
--use_dataset_cross_entropy_stopping \
--use_tqdm
if (( $global_idx % $jobs_per_gpu == 0 )); then
    wait
fi
done
done
done
done
done
done
done
wait
