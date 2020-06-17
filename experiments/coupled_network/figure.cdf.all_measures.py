import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns

from sys import argv

from common import get_complexity_measures, hoeffding_weight, get_hps, load_data, pretty_measure, sign_error


DATA_PATH = "../../data/nin_adjusted.csv"
ENVIRONMENT_CACHE_PATH = "./environment_cache"


def get_all_losses(data, hp, env_weights, env_losses, hps, hp_combo_id, min_weight=20.):
    """
    Gather the loss in each environment where some hyperparameter is varied from v1 to v2 and the remaining HPs are kept
    fixed. In each environment, the loss is given by the expected loss of importance sampling estimator over all random
    seeds.

    """
    hp_idx = hps.index(hp)

    # Unique values for the HP
    values = np.unique(data[hp]).tolist()

    all_losses = []

    for i, v1 in enumerate(values):
        # All points where Hi = v1
        v1_combos = [h for h in hp_combo_id if h[hp_idx] == v1]

        for j, v2 in enumerate(values):
            if v1 == v2:
                continue

            # Generate the coupled hp combos
            v2_combos = [v1c[: hp_idx] + (v2,) + v1c[hp_idx + 1:] for v1c in v1_combos]

            # Filter out v1_combos for which the v2_combo doesn't exist (e.g., job didn't finish)
            v1_combos_, v2_combos_, v1_idx, v2_idx = zip(*[(v1c, v2c, hp_combo_id[v1c], hp_combo_id[v2c]) for v1c, v2c
                                                           in zip(v1_combos, v2_combos) if v2c in hp_combo_id])

            weights = env_weights[v1_idx, v2_idx]
            losses = env_losses[v1_idx, v2_idx]

            # Filter out envs for which weight sum <= threshold
            selector = weights >= min_weight
            weights = weights[selector]
            losses = losses[selector]
            all_losses += losses.tolist()

    return np.array(all_losses)


def make_figure(datasets, tau=0, min_weight=10):
    data_key = "_".join(datasets)

    data = load_data(DATA_PATH)
    data = data.loc[[r["hp.dataset"] in datasets for _, r in data.iterrows()]]  # Select based on dataset

    extra_info = pickle.load(open(ENVIRONMENT_CACHE_PATH + "/env_losses__tau%d__%s.extra_info.pkl" % (tau, data_key), 
                                  "rb"))
    env_losses = pickle.load(open(ENVIRONMENT_CACHE_PATH + "/env_losses__tau%d__%s.pkl" % (tau, data_key), "rb"))
    env_weights = pickle.load(open(ENVIRONMENT_CACHE_PATH + "/env_weights__tau%d__%s.pkl" % (tau, data_key), "rb"))

    # Get the losses for each complexity measure, per hp
    complexity_losses_per_hp = {}
    for c in env_losses:
        complexity_losses_per_hp[c] = {}
        for hp in extra_info["hps"]:
            complexity_losses_per_hp[c][hp] = get_all_losses(data, hp, env_weights=env_weights,
                                                             env_losses=env_losses[c],
                                                             hp_combo_id=extra_info["hp_combo_id"],
                                                             hps=extra_info["hps"],
                                                             min_weight=min_weight)
        complexity_losses_per_hp[c]["all"] = np.hstack([complexity_losses_per_hp[c][h] for h in
                                                        complexity_losses_per_hp[c]])

    ordered_measures = np.array(list(env_losses.keys()))[np.argsort([np.mean(complexity_losses_per_hp[c]["all"]) for c
                                                                     in env_losses])].tolist()  # Order by mean
    # ordered_measures = np.sort(list(env_losses.keys())).tolist()  # Order by name

    ordered_measures.remove("complexity.l2_adjusted1")
    ordered_measures.remove("complexity.l2_dist_adjusted1")
    ordered_hps = ["all", "hp.lr", "hp.model_depth", "hp.model_width", "hp.train_size", "hp.dataset"]
    pretty_hps = {"all": "All", "hp.lr": "LR", "hp.model_depth": "Depth", "hp.model_width": "Width",
                  "hp.train_size": "Train size", "hp.dataset": "Dataset"}

    bins = np.linspace(0, 1, 100)
    f, axes = plt.subplots(ncols=1, nrows=len(ordered_hps), sharex=True, sharey=True)
    cbar_ax = f.add_axes([.91, .127, .02, .75])
    for ax, hp in zip(axes, ordered_hps):
        z = np.zeros((len(bins), len(ordered_measures)))
        for i, c in enumerate(ordered_measures):
            # Get losses
            losses = complexity_losses_per_hp[c][hp]

            # Plot mean and max
            ax.axvline(i, linestyle="-", color="white")

            if len(losses) > 0:
                ax.plot([i, i + 1], [np.mean(losses) * 100, np.mean(losses) * 100], color="orange", zorder=1)
                ax.plot([i, i + 1], [np.max(losses) * 100, np.max(losses) * 100], color="black", zorder=1)

                # Calculate CDF
                for j, b in enumerate(bins):
                    z[j, i] = (losses <= b).sum() / len(losses)
            else:
                ax.scatter([i + 0.5], [50], marker="x", color="red")

        if z.sum() > 0:
            heatmap = sns.heatmap(z, cmap="Blues_r", vmin=0.5, vmax=1, rasterized=True, ax=ax, cbar_ax=cbar_ax)
            heatmap.collections[0].colorbar.ax.tick_params(labelsize=8)

        ax.invert_yaxis()
        ax.set_yticks([0, 50, 100])
        ax.set_yticklabels([0, 0.5, 1], fontsize=6, rotation=0)
        ax.set_ylabel(pretty_hps[hp] + "\n(%d)" %
                      (len(complexity_losses_per_hp[list(complexity_losses_per_hp.keys())[0]][hp])), fontsize=8)
        ax.set_ylim(0, 101)
        
    axes[-1].set_xticks(np.arange(len(ordered_measures)) + 0.5)
    axes[-1].set_xticklabels([pretty_measure(c) for c in ordered_measures], rotation=45, fontsize=8, ha="right")
    f.set_size_inches(w=10, h=4.8)
    plt.savefig("figure__signerror_cdf_per_hp__ds_%s__tau_%d__mw_%f_cdf_per_hp.pdf" % (data_key, tau, min_weight), 
                bbox_inches="tight")


if __name__ == "__main__":
    dataset = argv[1]
    available_datasets = ["cifar10", "svhn"]
    if dataset not in available_datasets + ["all"]:
        raise ValueError("Invalid dataset specified.")
    elif dataset == "all":
        datasets = available_datasets
    else:
        datasets = [dataset]

    tau = float(argv[2])
    assert tau >= 0

    min_weight = float(argv[3])
    assert min_weight >= 0

    make_figure(datasets=datasets, tau=tau, min_weight=min_weight)
