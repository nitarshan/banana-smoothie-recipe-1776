import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns

from sys import argv

from common import get_complexity_measures, hoeffding_weight, get_hps, load_data, pretty_measure, sign_error


DATA_PATH = "../../data/nin_adjusted.csv"
ENVIRONMENT_CACHE_PATH = "./environment_cache"


pretty_hps = {"all": "All", "hp.lr": "Learning rate", "hp.model_depth": "Depth", "hp.model_width": "Width",
              "hp.train_size": "TS", "hp.dataset": "Dataset"}


def triangle_cdf_plots_get_losses(data, hp, env_weights, env_losses, hps, hp_combo_id, min_weight=20.):
    """
    Gather the loss in each environment where some hyperparameter is varied from v1 to v2 and the remaining HPs are kept
    fixed. In each environment, the loss is given by the expected loss of importance sampling estimator over all random
    seeds.

    """
    hp_idx = hps.index(hp)

    # Unique values for the HP
    values = np.unique(data[hp])

    box_losses = {}

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
            box_losses[(v1, v2)] = losses

    return box_losses


def make_figure(datasets, measure, hp, tau=0, min_weight=10):
    data_key = "_".join(datasets)

    data = load_data(DATA_PATH)
    data = data.loc[[r["hp.dataset"] in datasets for _, r in data.iterrows()]]  # Select based on dataset

    extra_info = pickle.load(open(ENVIRONMENT_CACHE_PATH + "/env_losses__tau%d__%s.extra_info.pkl" % (tau, data_key),
                                  "rb"))
    env_losses = pickle.load(open(ENVIRONMENT_CACHE_PATH + "/env_losses__tau%d__%s.pkl" % (tau, data_key), "rb"))
    env_weights = pickle.load(open(ENVIRONMENT_CACHE_PATH + "/env_weights__tau%d__%s.pkl" % (tau, data_key), "rb"))

    box_losses = triangle_cdf_plots_get_losses(data, hp, env_weights=env_weights,
                                               env_losses=env_losses[measure],
                                               hp_combo_id=extra_info["hp_combo_id"], hps=extra_info["hps"],
                                               min_weight=min_weight)

    values = np.unique([vals[0] for vals in box_losses])

    f, axes = plt.subplots(ncols=len(values), nrows=len(values), sharex=True, sharey=True)
    cbar_ax = f.add_axes([.77, .127, .05, .55])
    for i, v1 in enumerate(values):
        for j, v2 in enumerate(values):
            if j >= i:
                axes[i, j].set_visible(False)
                continue

            bins = np.linspace(0, 1, 100)

            # Add stats markers
            if len(box_losses[(v1, v2)]) != 0:
                # Calculate CDF
                z = np.zeros((len(bins), 1))
                for k, b in enumerate(bins):
                    z[k] = (box_losses[(v1, v2)] <= b).sum() / len(box_losses[(v1, v2)])
                
                heatmap = sns.heatmap(z, cmap="Blues_r", vmin=0.5, vmax=1, rasterized=True, ax=axes[i, j],
                                      cbar_ax=cbar_ax)

                axes[i, j].axhline(np.mean(box_losses[(v1, v2)]) * 100, color="orange")
                axes[i, j].axhline(np.max(box_losses[(v1, v2)]) * 100, color="black")
            else:
                axes[i, j].scatter([0.5], [50], color="red", marker="x", s=30)

            axes[i, j].invert_yaxis()
            axes[i, j].set_ylim([-0.1, 100])
            axes[i, j].set_yticks([0, 50, 100])
            axes[i, j].set_yticklabels([0, 0.5, 1], fontsize=5, rotation=0)
            axes[i, j].set_xticklabels([], fontsize=6, rotation=0)
            axes[i, j].xaxis.set_visible(False)
            axes[i, j].yaxis.set_visible(False)

    for i, v1 in enumerate(values):
        axes[i, 0].set_ylabel("%s=%d" % (pretty_hps[hp], v1), fontsize=6)
        axes[i, 0].yaxis.set_visible(True)
        axes[-1, i].set_xlabel("%s=%d" % (pretty_hps[hp], v1), fontsize=6, rotation=45, ha="right")
        axes[-1, i].xaxis.set_visible(True)

    cbar = heatmap.collections[0].colorbar.ax.tick_params(labelsize=6)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    f.set_size_inches(w=1.3, h=2.7)
    plt.savefig("figure_triangle_cdf__ds_%s__tau_%d__mw_%f_gm_%s_hp_%s.pdf" % (data_key, tau, min_weight,
                                                                               pretty_measure(measure), hp),
                bbox_inches="tight")


if __name__ == "__main__":
    dataset = argv[1]
    available_datasets = ["cifar10", "svhn"]
    if dataset not in available_datasets + ["all"]:
        raise ValueError("Invalid dataset specificed.")
    elif dataset == "all":
        datasets = available_datasets
    else:
        datasets = [dataset]

    measure = argv[2]
    # TODO: validate measures

    hp = argv[3]
    available_hps = ["hp.model_depth", "hp.model_width", "hp.lr"]
    if hp not in available_hps:
        raise ValueError("Invalid hyperparameter (hp) specified.")

    tau = float(argv[4])
    assert tau >= 0

    min_weight = float(argv[5])
    assert min_weight >= 0

    make_figure(datasets=datasets, measure=measure, hp=hp, tau=tau, min_weight=min_weight)
