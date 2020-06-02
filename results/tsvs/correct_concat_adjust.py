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

def preprocess(dataset):
  df = []
  for dataset_size in [6_250, 12_500, 25_000, 50_000]:
    df.append(pd.read_csv(f'./results/tsvs/nin_{dataset}.{dataset_size}.tsv', sep='\t', index_col=0))
  df = pd.concat(df)
  if dataset == 'cifar10':
    df = correct_sum_norms(df)
  df['hp.dataset'] = dataset
  df = adjust_measures(df)
  df['hp.train_dataset_size'] = df['train_dataset_size']
  df['complexity.log_spec_orig_main'] = df['complexity.log_prod_of_spec_over_margin'] + np.log(df['complexity.fro_over_spec'])
  return df

# SVHN + CIFAR-10
df1 = preprocess('cifar10')
df2 = preprocess('svhn')
df3 = pd.concat([df1, df2])

# Save to CSVs
df1.to_csv(f'./results/tsvs/nin_cifar_adjusted.csv', float_format="%.4g")
df2.to_csv(f'./results/tsvs/nin_svhn_adjusted.csv', float_format="%.4g")
df3.to_csv(f'./results/tsvs/nin_adjusted.csv', float_format="%.4g")

# Test CSVs
def test_num_envs(df):
  assert(len(df['hp.lr'].unique())==5)
  assert(len(df['hp.model_depth'].unique())==5)
  assert(len(df['hp.model_width'].unique())==5)

test_num_envs(pd.read_csv(f'./results/tsvs/nin_cifar_adjusted.csv'))
test_num_envs(pd.read_csv(f'./results/tsvs/nin_svhn_adjusted.csv'))
test_num_envs(pd.read_csv(f'./results/tsvs/nin_adjusted.csv'))
