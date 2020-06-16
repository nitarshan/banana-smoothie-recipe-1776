import numpy as np
import pandas as pd


TARGET_CE = 0.01
TARGET_TOL = 1e-6

BLACKLISTED_MEASURES = {'complexity.l2', 'complexity.l2_dist'}
# Params is deterministic

DATASIZES = (50_000, 25_000, 12_500, 6_250)


def average_over_repeats(data):
    """
    Take the expectation of generalization measures and errors over all points with the same hyperparameters

    """
    return data.groupby("experiment_id").mean()


def get_complexity_measures(data):
    return [c for c in data.columns if c.startswith("complexity.") and c not in BLACKLISTED_MEASURES]


def get_hps(data):
    return [c for c in data.columns if c.startswith("hp.")]


def get_gen_measures(data):
    return [c for c in data.columns if c.startswith("gen.") and c != "gen.train_acc"]


def hoeffding_weight(delta_risk, n=10000, shift=0):
    """
    This value has the following guarantee. If your measurement of the risk is computed
    using n (say n=10,000) independent samples, then accepting samples only when this
    value > p would mean that those samples are legit different with probability at least p

    Parameters:
    -----------
    delta_risk: float
        The absolute difference between two estimated risks
    n: int
        The size of the data sample used to estimate the risks

    Returns:
    --------
    weight: float
        Probability that the two risks are actually different

    """
    def phi(x, n):
        return 2 * np.exp(-2 * n * (x / 2)**2)
    return max(0., 1. - phi(max(np.abs(delta_risk) - shift, 0), n))**2


def load_data(data_path):
    def clean_data(data, name=''):
        # Discard measurements that do not meet crossentropy standards and warn
        # These might have reached the max number of epochs.
        n_before = data.shape[0]
        data = data.loc[data["is.converged"]]
        if data.shape[0] < n_before:
            print(f"[{name}] Warning: discarded %d results that did not meet the cross-entropy standards." % (n_before - data.shape[0]))

        # Discard measurements that do not meet accuracy standards and warn
        n_before = data.shape[0]
        data = data.loc[data["is.high_train_accuracy"]]
        if data.shape[0] < n_before:
            print(f"[{name}] Warning: discarded %d results that did not meet the accuracy standards." % (n_before - data.shape[0]))

        return data

    data = clean_data(pd.read_csv(data_path, index_col=0), "data")
    # Some needed preprocessing
    data["hp.train_size"] = data.train_dataset_size
    data["hp.lr"] = data["hp.lr"].round(4)  # Needed since some runs were computed accross difference devices
    # data["hp.dataset"] = data_name.upper()
    del data["hp.train_dataset_size"]
    return data


def sign_error(c1, c2, g1, g2):
    """
    This loss function measures the positive association between two data points ranked
    according to two scores c and g, i.e., a generalization measure and the generalization
    gap, respectively.

    Parameters:
    -----------
    c1, c2: float
        The value of the generalization measure for the two data points
    g1, g2: float
        The value of the generalization gap for the two data points

    Returns:
    --------
    loss: float
        This loss is bounded between 0 and 1. A value of 0 indicates that the points are
        ranked equally according to c and g. A positive value indicates that the rankings
        did not match.

    """
    error = float(np.sign(c1 - c2) * np.sign(g1 - g2))
    return (1 - error) / 2


def pretty_measure(c):
    """
    Pretty print measure names

    """
    if c == "complexity.params":
        c = "complexity.num.params"
    return c.replace("complexity.", "").replace("_", ".").replace(".adjusted1", "").replace("log.", "")
