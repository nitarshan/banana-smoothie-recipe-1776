# %% Imports
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# %% Flags
parser = argparse.ArgumentParser(description='Export Wandb results')
parser.add_argument('--tag', type=str)
flags = parser.parse_args()
tag = flags.tag

# %% Helper functions
def get_model_results(results):
    env_split_mode = {}
    for split, split_results in results.groupby("env_split"):
        env_split_mode[split] = split_results[["actual_measure", "risk_max"]]
    return env_split_mode

def get_best_by_measure(data):
    data = pd.DataFrame(data).reset_index()
    measures = [c for c in data.actual_measure.unique() if len(c.split(".")) == 2]
    data["measure"] = data.actual_measure.apply(lambda x: ".".join(x.split(".")[:2]))
    data = data.iloc[data.groupby("measure").risk_max.idxmin()]
    return data

def preprocess_columns(data):
    data["mae_max"] = np.sqrt(data.risk_max)
    data["pretty_measure"] = [c.replace("complexity.", "").replace("_adjusted1", "").replace("_", "-") for c in data.actual_measure]
    return data

def subtract_baseline(data, baseline_mae):
    data["mae_max_vs_baseline"] = baseline_mae - data["mae_max"]
    return data

# %% Load data
print(tag)

resultspath = Path(f'results/regression/')
resultspath.mkdir(exist_ok=True)

df = pd.read_csv(resultspath / f'{tag}_export.csv')[['lr', 'bias', 'datafile', 'env_split', 'actual_measure', 'only_bias__ignore_input', 'selected_single_measure', 'bias.1', 'loss', 'weight', '_runtime', 'risk_max', 'risk_min', 'train_mse', 'risk_range', 'robustness_penalty']]
affine = get_model_results(df[(df['bias']==True) & (df['only_bias__ignore_input']==False)].copy())
weight_only = get_model_results(df[(df['bias']==False) & (df['only_bias__ignore_input']==False)].copy())
bias_only = get_model_results(df[(df['bias']==True) & (df['only_bias__ignore_input']==True)].copy())

# %% Plot regression results
sns.set_style("darkgrid", {'xtick.bottom': True})
plotpath = Path(f'plots/{tag}/')
plotpath.mkdir(parents=True, exist_ok=True)
order = None

for idx, split in enumerate(sorted(df.env_split.unique())):
    plt.figure(figsize=(8,1.5))
    baselines_bias = {split: preprocess_columns(bias_only[split]).mae_max.values[0] for split in bias_only}
    plot_results = preprocess_columns(get_best_by_measure(affine[split]))
    plot_results = subtract_baseline(plot_results, baseline_mae=baselines_bias[split])
    order = plot_results.sort_values("mae_max_vs_baseline", ascending=False).pretty_measure if order is None else order
    sns.barplot(data=plot_results, order=order, x="pretty_measure", y="mae_max", palette="deep")
    plt.axhline(baselines_bias[split], label='bias-only baseline')
    plt.xticks(rotation=90)
    #plt.title(f'split {split}')
    plt.legend(loc='lower right')
    plt.xlabel('Generalization Measure')
    plt.ylabel('Robust RMSE')
    plt.tight_layout()
    plt.xticks(rotation=45,ha='right')
    plt.yticks(fontsize=8)
    plt.xticks(fontsize=8)
    plt.savefig(plotpath / f'{split}_mae_all_vs_baseline.pdf', bbox_inches='tight')
    plt.close()
