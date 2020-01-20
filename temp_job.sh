# 1. Load your environment
echo 'Load Environment'
source /network/home/rajkuman/.bashrc
module purge
module load anaconda/3
source $CONDA_ACTIVATE
conda activate ccm

# 2. Prepare directories and copy dataset onto the compute node
echo 'Loading Datasets'
mkdir -p /network/tmp1/rajkuman/results
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

python run_experiment.py single \
--root_dir=$SLURM_TMPDIR \
--model_type='DEEP' \
--model_width=30 \
--model_depth=2 \
--dataset_type='MNIST' \
--optimizer_type='ADAM' \
--lr=0.01 \
--epochs=200 \
--batch_size=100 \
--complexity_type='L2' \
--complexity_lambda=None \
--lagrangian_type='AUGMENTED' \
--lagrangian_target=3 \
--lagrangian_start_epoch=0 \
--lagrangian_start_mu=1e-6 \
--lagrangian_tolerance=0.01 \
--lagrangian_patience_batches=500 \
--lagrangian_improvement_rate=0.75 \
--lagrangian_start_lambda=0 \
--lagrangian_convergence_tolerance=1e-3 \
--global_convergence_tolerance=5e-6 \
--global_convergence_patience_windows=2 \
--comet_api_key=$COMET_API_KEY \
--comet_tag='temp' \
--use_cuda
