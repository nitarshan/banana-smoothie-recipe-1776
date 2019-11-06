# Causal Complexity Measures

Hello! To make use of this repository you need to first set up the conda environment for it:
1) `conda env create -f environment.yml`
2) `source activate causal_complexity_measures`

Look at `deep_mnist_l2_experiment` in `run_experiment.py` to see how an experiment can be launched (in parallel across CPUs), and look at `experiment_config.py` for experiment options. You can modify `run_experiment.py` as needed (or create a new one for a new experiment), and can launch the experiment with `./run_experiment.py`. (You might need to chmod that file or your new one to allow it to run as an executable).

After running an experiment, you can share the tensorboard logs via http://tensorboard.dev to a public link.
