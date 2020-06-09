# https://stackoverflow.com/a/42098494
function parallel {
  local time1=$(date +"%H:%M:%S")
  local time2=""

  echo "($time1) $@"
  "$@" && time2=$(date +"%H:%M:%S") && echo "finishing ($time1 -- $time2)..." &

  local my_pid=$$
  local children=$(ps -eo ppid | grep -w $my_pid | wc -w)
  children=$((children-1))
  if [[ $children -ge $max_jobs ]]; then
    wait -n
  fi
}

measures="log_spec_orig_main path_norm_over_margin path_norm param_norm pacbayes_orig pacbayes_mag_orig pacbayes_mag_init pacbayes_mag_flatness pacbayes_init pacbayes_flatness log_sum_of_spec_over_margin log_sum_of_spec log_sum_of_fro_over_margin log_sum_of_fro log_prod_of_spec_over_margin log_prod_of_spec log_prod_of_fro_over_margin log_prod_of_fro inverse_margin fro_over_spec fro_dist dist_spec_init"
max_jobs=$(nproc)
steps=10000
lr=0.005

envs='all lr width depth train_size'
exp_types='v1 v2 v3'
wandb_tag='june5'

for exp_type in $exp_types; do
  for env in $envs; do
    if [ "$exp_type" != "v1" ] || [ "$env" == "all" ]; then
      for measure in $measures; do
        parallel python regression.py \
        --env_split=$env \
        --exp_type=$exp_type \
        --selected_single_measure=$measure \
        --bias=True \
        --lr=$lr \
        --steps=$steps \
        --wandb_tag=${wandb_tag}_$exp_type
      done
      # Bias only baseline
      parallel python regression.py \
      --env_split=$env \
      --exp_type=$exp_type \
      --only_bias__ignore_input=True \
      --bias=True \
      --lr=$lr \
      --steps=$steps \
      --wandb_tag=${wandb_tag}_$exp_type
    fi
  done

  wait
  echo 'done '$exp_type

  wandb sync
  python results/export_regression_results.py --tag=${wandb_tag}_$exp_type
  python plots/plot_regression_results.py --tag=${wandb_tag}_$exp_type
done
