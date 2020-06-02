import pandas as pd
import numpy as np


def correct_sum_norms(df):
  sum_measures = ['complexity.log_sum_of_fro', 'complexity.log_sum_of_spec', 'complexity.log_sum_of_fro_over_margin', 'complexity.log_sum_of_spec_over_margin']
  d = df[['hp.model_depth']] * 6 + 2
  df[sum_measures] += np.log(d.values) - d.values 
  return df


def adjust_measures(df):
  measures = [col for col in df.columns if col.startswith('complexity.') and '.params' not in col]
  adjust1 = [col for col in measures if '.log' not in col]
  adjust1_log = [col for col in measures if '.log' in col]

  adjust1_new = [x+'_adjusted1' for x in adjust1]
  adjust1_log_new = [x+'_adjusted1' for x in adjust1_log]

  m = df[['train_dataset_size']].values

  df[adjust1_new] = np.sqrt(df[adjust1] / m)
  df[adjust1_log_new] = 0.5 * (df[adjust1_log] - np.log(m))

  return df


# CIFAR-10
df1 = []
for dataset_size in [6_250, 12_500, 25_000, 50_000]:
  df1.append(pd.read_csv(f'./results/tsvs/nin_cifar10.{dataset_size}.tsv', sep='\t', index_col=0))
df1 = pd.concat(df1)
df1 = correct_sum_norms(df1)
df1['hp.dataset'] = 'CIFAR10'

# SVHN
df2 = []
for dataset_size in [6_250, 12_500, 25_000, 50_000]:
  df2.append(pd.read_csv(f'./results/tsvs/nin_svhn.{dataset_size}.tsv', sep='\t', index_col=0))
df2 =  pd.concat(df2)
df2['hp.dataset'] = 'SVHN'

# SVHN + CIFAR-10
df3 = pd.concat([df1, df2])
df2 = adjust_measures(df2)
df3['hp.train_dataset_size'] = df3['train_dataset_size']
df3['complexity.log_spec_orig_main'] = df3['complexity.log_prod_of_spec_over_margin'] + np.log(df3['complexity.fro_over_spec'])

# Save to CSVs
df3.to_csv(f'./results/tsvs/nin_adjusted.csv', float_format="%.4g")

df3 = pd.read_csv(f'./results/tsvs/nin_adjusted.csv')
assert(len(df3['hp.lr'].unique())==5)
assert(len(df3['hp.model_depth'].unique())==5)
assert(len(df3['hp.model_width'].unique())==5)