import pandas as pd
import numpy as np

# Correct for incorrect measure calculations
measures = ['complexity.log_sum_of_fro', 'complexity.log_sum_of_spec', 'complexity.log_sum_of_fro_over_margin', 'complexity.log_sum_of_spec_over_margin']
for dataset_size in [6_250, 12_500, 25_000, 50_000]:
  df = pd.read_csv(f'./results/tsvs/nin_cifar10.{dataset_size}.tsv', sep='\t', index_col=0)
  effective_depths = df[['hp.model_depth']] * 6 + 2
  df[measures] += np.log(effective_depths.values) - effective_depths.values 
  df.to_csv(f'./results/tsvs/nin_cifar10.{dataset_size}_corrected.tsv', sep='\t', float_format="%g")

# Concatenate all dataset size tsv's into a single tsv
dfs = []
for dataset_size in [6_250, 12_500, 25_000, 50_000]:
  dfs.append(pd.read_csv(f'./results/tsvs/nin_cifar10.{dataset_size}_corrected.tsv', sep='\t', index_col=0))
pd.concat(dfs).to_csv(f'./results/tsvs/nin_cifar10_corrected.tsv', sep='\t', float_format="%g")

# Adjust measures accounting for dataset size
df = pd.read_csv('./results/tsvs/nin_cifar10_corrected.tsv', sep='\t', index_col=0)

print(df.columns)

adjust1 = [col for col in df.columns if col.startswith('complexity.') and '.log' not in col and '.params' not in col and '.inverse_margin' not in col]
adjust1_log = [col for col in df.columns if col.startswith('complexity.') and '.log' in col and '.params' not in col and '.inverse_margin' not in col]
adjust2 = [col for col in df.columns if col.startswith('complexity.') and 'path_norm' in col]

adjust1_new = [x+'_adjusted1' for x in adjust1]
adjust1_log_new = [x+'_adjusted1' for x in adjust1_log]
adjust2_new = [x+'_adjusted2' for x in adjust2]
print('adjust1', adjust1)
print('adjust1_log', adjust1_log)
print('adjust2', adjust2)

d = df[['hp.model_depth']].values * 6 + 2
m = df[['train_dataset_size']].values

df[adjust1_new] = np.sqrt(df[adjust1] / m)
df[adjust1_log_new] = 0.5 * (df[adjust1_log] - np.log(m))
df[adjust2_new] = np.sqrt(((df[adjust2] / np.sqrt(d)) ** (2 * d)) / m)

df.to_csv(f'./results/tsvs/nin_cifar10_corrected_adjusted.tsv', sep='\t', float_format="%g")
