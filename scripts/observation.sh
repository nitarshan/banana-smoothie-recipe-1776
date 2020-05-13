# 1. Load your environment
echo 'Load Environment'
: '
source ~/.bashrc
module purge
module load anaconda/3
source $CONDA_ACTIVATE
conda activate base
conda activate ccm
'

# 2. Prepare directories and copy dataset onto the compute node
echo "Loading Datasets"
mkdir -p /network/tmp1/$USER/results
mkdir -p $SLURM_TMPDIR/data/MNIST/
mkdir -p $SLURM_TMPDIR/results/
cp -r /network/data1/mnist/processed $SLURM_TMPDIR/data/MNIST/
cp -r /network/data1/cifar/cifar-10-batches-py $SLURM_TMPDIR/data/
# cp -r /network/data1/svhn $SLURM_TMPDIR/data/
# 3. Launch your job
echo "Launching Experiment"
COMET_API_KEY="a21c3c978d64ab77bd9b9329f8640b46edaa9ec0"
#tag="tedros25"
#tag="interventional_path_norm_ethan"
#tag="interventional_path_norm_ethan__more_epochs"
#tag="interventional_path_norm_ethan__correct"
#tag="interventional_path_norm_ethan__actually_correct"
#tag="interventional_path_norm_ethan__weird"
#tag="interventional_path_norm_ethan__more_correct"
tag="interventional_path_norm_ethan__agi"
model=DEEP
dataset=MNIST
optims="SGD_MOMENTUM"
#measures="PATH_NORM"
measures="L2"
#measures="PATH_NORM_OVER_MARGIN"
targets="15"
#targets="10 30"
#lrs="0.01"
lrs="0.001"
widths="100"
depths="3"
global_idx=0
jobs_per_gpu=3
for measure in $measures; do
for target in $targets; do
for optim in $optims; do
for lr in $lrs; do
for width in $widths; do
for depth in $depths; do
let "global_idx++"
python run_experiment.py single \
--root_dir=$SLURM_TMPDIR \
--model_type=DEEP \
--model_depth=$depth \
--model_width=$width \
--dataset_type=MNIST \
--optimizer_type=$optim \
--lr=$lr \
--epochs=150 \
--batch_size=128 \
--complexity_type=NONE \
--complexity_lambda=None \
--lagrangian_type=NONE \
--lagrangian_target=$target \
--lagrangian_start_epoch=0 \
--lagrangian_start_mu=1e-1 \
--lagrangian_tolerance=0.1 \
--lagrangian_patience_batches=200 \
--lagrangian_improvement_rate=0.75 \
--lagrangian_start_lambda=0 \
--global_convergence_method=leq \
--lagrangian_convergence_tolerance=1e-4 \
--global_convergence_tolerance=1e-8 \
--global_convergence_patience=30 \
--global_convergence_target=0.01 \
--comet_api_key=$COMET_API_KEY \
--log_epoch_freq=10 \
--comet_tag=$tag \
--use_cuda=False \
--use_wandb=True
if (( $global_idx % $jobs_per_gpu == 0 )); then
    wait
fi
done
done
done
done
done
done
wait