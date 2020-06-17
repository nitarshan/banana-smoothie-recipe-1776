create-environments: .require_dataset .require_tau
	python3 create_environments.py $(dataset) $(tau)

figure-cdf-all-measures: .require_dataset .require_tau .require_min_weight
figure-cdf-all-measures: environment_cache/env_losses__tau$(tau)__$(dataset).pkl
figure-cdf-all-measures: environment_cache/env_losses__tau$(tau)__$(dataset).extra_info.pkl
figure-cdf-all-measures: environment_cache/env_weights__tau$(tau)__$(dataset).pkl
	python3 figure.cdf.all_measures.py $(dataset) $(tau) $(min_weight)

.require_dataset:
ifndef dataset
	$(error dataset parameter is required. Add dataset=[cifar10, svhn, all] to your command.)
endif

.require_tau:
ifndef tau
	$(error tau parameter is required. The value should be in percentage points (e.g., 1, not 0.01). Add tau=<value> to your command.)
endif

.require_min_weight:
ifndef min_weight
	$(error min_weight parameter is required. This is the minimum sum of Hoeffding weights in an environment for it to be considered in the analysis. Add min_weight=<value> to your command.)
endif