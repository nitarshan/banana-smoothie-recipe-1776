# 1. Load your environment
echo 'Load Environment'
source /network/home/$USER/.bashrc
module purge
module load anaconda/3
source $CONDA_ACTIVATE
conda activate base
conda activate ccm

# 2. Prepare directories and copy dataset onto the compute node
echo 'Loading Datasets'
mkdir -p /network/tmp1/$USER/results
mkdir -p $SLURM_TMPDIR/data/MNIST/
mkdir $SLURM_TMPDIR/results
cp -r /network/data1/mnist/processed $SLURM_TMPDIR/data/MNIST/
cp -r /network/data1/cifar/cifar-10-batches-py $SLURM_TMPDIR/data/
# cp -r /network/data1/svhn $SLURM_TMPDIR/data/

# 3. Launch your job
echo 'Launching Experiment'
model='DEEP'
dataset='MNIST'
optimizer='SGD_MOMENTUM'
measures='L2 PROD_OF_FRO SUM_OF_FRO PARAM_NORM PATH_NORM'
targets=(19.73 8096.64 26.50 381.56 45.56) # See complexity_lambda_analysis.ipynb
lrs='0.05'
global_idx=0
jobs_per_gpu=4

for lr in $lrs; do
let "global_idx++"
python run_experiment.py single \
--root_dir=$SLURM_TMPDIR \
--model_type='RESNET' \
--model_depth=2 \
--model_width=1 \
--dataset_type='CIFAR10' \
--optimizer_type='SGD_MOMENTUM' \
--lr=$lr \
--epochs=500 \
--batch_size=128 \
--complexity_type='NONE' \
--complexity_lambda=None \
--lagrangian_type='NONE' \
--lagrangian_target=3 \
--lagrangian_start_epoch=0 \
--lagrangian_start_mu=1e-6 \
--lagrangian_tolerance=0.01 \
--lagrangian_patience_batches=200 \
--lagrangian_improvement_rate=0.75 \
--lagrangian_start_lambda=0 \
--global_convergence_method='leq' \
--lagrangian_convergence_tolerance=1e-4 \
--global_convergence_tolerance=1e-8 \
--global_convergence_patience=30 \
--global_convergence_target=0.01 \
--comet_api_key=$COMET_API_KEY \
--comet_tag='lr_test_7' \
--log_epoch_freq=10 \
--use_cuda
if (( $global_idx % $jobs_per_gpu == 0 )); then
    wait
fi
done
wait