# Causal Complexity Measures

Hello! To make use of this repository you need to first set up the conda environment for it:
1) `conda env create -f environment.yml`
2) `conda activate rgm`

Look at `single` and `multi` in `run_experiment.py` to see how an experiment can be launched (`multi` handles parallelizing of CPU experiments within python), and look at `experiment_config.py` for experiment options. For calling these methods via command-line or for usage on SLURM, see a simple example script in `submit_job.sh`.

After running an experiment, you can share the tensorboard logs via http://tensorboard.dev to a public link.

## Tests

You can run all tests by running the following command from the project directory:

```
python -m unittest discover -s tests
```