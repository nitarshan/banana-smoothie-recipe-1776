import numpy as np
import pickle

from os import makedirs
from sys import argv
from tqdm import tqdm

from common import get_complexity_measures, hoeffding_weight, get_hps, load_data, sign_error


DATA_PATH = "../data/nin_adjusted.csv"
SAVE_PATH = "environment_cache"
makedirs(SAVE_PATH, exist_ok=True)


def create_environments(datasets, tau):
    data = load_data(DATA_PATH)
    data = data.loc[[r["hp.dataset"] in datasets for _, r in data.iterrows()]]  # Select based on dataset

    # List of hyperparameters
    hps = get_hps(data)

    # Assign a unique index to each HP combination
    hp_combo_id = set([tuple(row[hps]) for _, row in data.iterrows()])
    hp_combo_id = dict(zip(hp_combo_id, range(len(hp_combo_id))))

    # Extract all complexity measures and the generalization gap for each HP combo
    hp_c = [(tuple(row[hps].values), ({c: row[c] for c in get_complexity_measures(data)}, row["gen.gap"])) for _, row
            in data.iterrows()]

    c_measures = [c for c in get_complexity_measures(data) if ("adjusted1" in c or c + "_adjusted1" not in
                                                               get_complexity_measures(data)) and "adjusted2" not in c]

    env_losses = {c: np.zeros((len(hp_combo_id), len(hp_combo_id))) for c in c_measures}
    env_weights = np.zeros((len(hp_combo_id), len(hp_combo_id)))

    for h1, (all_c1, g1) in tqdm(hp_c, position=0, leave=True):
        h1_id = hp_combo_id[h1]
        for h2, (all_c2, g2) in hp_c:
            h2_id = hp_combo_id[h2]
            if h2_id == h1_id:
                continue

            # These are independent of the complexity measure
            weight = hoeffding_weight(np.abs(g1 - g2), n=10000, shift=float(tau) / 100)
            env_weights[h1_id, h2_id] += weight

            # Get the loss for each complexity measure
            for c in c_measures:
                c1 = all_c1[c]
                c2 = all_c2[c]
                env_losses[c][h1_id, h2_id] += sign_error(c1, c2, g1, g2) * weight

    for c in c_measures:
        mask = np.isclose(env_weights, 0)
        env_losses[c][~mask] /= env_weights[~mask]
        env_losses[c][mask] = 0

    # Save to disk
    pickle.dump({
        "hp_combo_id": hp_combo_id,
        "hps": hps
    }, open(SAVE_PATH + "/env_losses__tau%d__%s.extra_info.pkl" % (tau, "_".join(datasets)), "wb"))
    pickle.dump(env_losses, open(SAVE_PATH + "/env_losses__tau%d__%s.pkl" % (tau, "_".join(datasets)), "wb"))
    pickle.dump(env_weights, open(SAVE_PATH + "/env_weights__tau%d__%s.pkl" % (tau, "_".join(datasets)), "wb"))


if __name__ == "__main__":
    dataset = argv[1]
    available_datasets = ["cifar10", "svhn"]
    if dataset not in available_datasets + ["all"]:
        raise ValueError("Invalid dataset specificed.")
    elif dataset == "all":
        datasets = available_datasets
    else:
        datasets = [dataset]

    tau = float(argv[2])
    assert tau >= 0

    create_environments(datasets=datasets, tau=tau)
