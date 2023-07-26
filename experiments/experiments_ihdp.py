import os
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from .data_utils import download_if_needed
from .experiment_utils import DataGenerator

DATA_DIR = "data/ihdp/"
IHDP_TRAIN_NAME = "ihdp_npci_1-100.train.npz"
IHDP_TEST_NAME = "ihdp_npci_1-100.test.npz"

TRAIN_URL = "https://www.fredjo.com/files/ihdp_npci_1-100.train.npz"
TEST_URL = "https://www.fredjo.com/files/ihdp_npci_1-100.test.npz"

IHDP_NUMERIC_COLS = [0, 1, 2, 3, 4, 5]
IHDP_N_NUMERIC_COLS = len(IHDP_NUMERIC_COLS)


# data utils -------------------------------------------------------------------
def load_data_npz(fname, get_po: bool = True):
    """Load data set (adapted from https://github.com/clinicalml/cfrnet)"""
    if fname[-3:] == "npz":
        data_in = np.load(fname)
        data = {"X": data_in["x"], "w": data_in["t"], "y": data_in["yf"]}
        try:
            data["ycf"] = data_in["ycf"]
        except Exception:  # pylint: disable=broad-exception-caught
            data["ycf"] = None
    else:
        raise ValueError("This loading function is only for npz files.")

    if get_po:
        data["mu0"] = data_in["mu0"]
        data["mu1"] = data_in["mu1"]

    data["HAVE_TRUTH"] = not data["ycf"] is None
    data["dim"] = data["X"].shape[1]
    data["n"] = data["X"].shape[0]

    return data


def get_one_data_set(D, i_exp, get_po: bool = True):
    """Get data for one experiment. Adapted from https://github.com/clinicalml/cfrnet"""
    D_exp = {}
    D_exp["X"] = D["X"][:, :, i_exp - 1]
    D_exp["w"] = D["w"][:, i_exp - 1 : i_exp]
    D_exp["y"] = D["y"][:, i_exp - 1 : i_exp]
    if D["HAVE_TRUTH"]:
        D_exp["ycf"] = D["ycf"][:, i_exp - 1 : i_exp]
    else:
        D_exp["ycf"] = None

    if get_po:
        D_exp["mu0"] = D["mu0"][:, i_exp - 1 : i_exp]
        D_exp["mu1"] = D["mu1"][:, i_exp - 1 : i_exp]

    return D_exp


def prepare_ihdp_data(data_train, data_test, setting="original", return_ytest=True):
    if setting == "original":
        X, y, w, mu0, mu1 = data_train["X"], data_train["y"], data_train["w"], data_train["mu0"], data_train["mu1"]

        X_t, y_t, w_t, mu0_t, mu1_t = data_test["X"], data_test["y"], data_test["w"], data_test["mu0"], data_test["mu1"]

    elif setting == "modified":
        X, y, w, mu0, mu1 = data_train["X"], data_train["y"], data_train["w"], data_train["mu0"], data_train["mu1"]

        X_t, y_t, w_t, mu0_t, mu1_t = data_test["X"], data_test["y"], data_test["w"], data_test["mu0"], data_test["mu1"]
        y[w == 1] = y[w == 1] + mu0[w == 1]
        mu1 = mu0 + mu1
        mu1_t = mu0_t + mu1_t
    else:
        raise ValueError("Setting should in [original, modified]")

    cate = mu1 - mu0
    cate_t = mu1_t - mu0_t

    if return_ytest:
        return X, y, w, mu0, mu1, cate, X_t, y_t, w_t, mu0_t, mu1_t, cate_t

    return X, y, w, mu0, mu1, cate, X_t, mu0_t, mu1_t, cate_t


def download_ihdp_data():
    data_path = Path(DATA_DIR)
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # download if needed
    train_csv = data_path / IHDP_TRAIN_NAME
    test_csv = data_path / IHDP_TEST_NAME

    download_if_needed(train_csv, http_url=TRAIN_URL)
    download_if_needed(test_csv, http_url=TEST_URL)


class IHDPDataGenerator(DataGenerator):
    # pylint: disable-next=super-init-not-called
    def __init__(self, setting="original", n_train=500):
        self.n_train = n_train
        self.setting = setting

        self.name = "ihdp-" + str(setting) + "-" + str(n_train)

    def __call__(self, seed=42):
        assert seed in range(100)

        download_ihdp_data()

        data_train = load_data_npz(DATA_DIR + IHDP_TRAIN_NAME, get_po=True)
        data_test = load_data_npz(DATA_DIR + IHDP_TEST_NAME, get_po=True)

        data_exp = get_one_data_set(data_train, i_exp=seed + 1, get_po=True)
        data_exp_test = get_one_data_set(data_test, i_exp=seed + 1, get_po=True)
        (  # pyright: ignore
            X,
            y,
            w,
            mu0,
            mu1,
            cate,
            X_test,
            y_test,
            w_test,
            mu_0_test,
            mu_1_test,
            cate_test,
        ) = prepare_ihdp_data(data_exp, data_exp_test, setting=self.setting)

        (
            X_train,
            X_val,
            w_train,
            w_val,
            y_train,
            y_val,
            mu0_train,
            mu0_val,
            mu1_train,
            mu1_val,
            cate_train,
            cate_val,
        ) = train_test_split(X, w, y, mu0, mu1, cate, train_size=self.n_train, random_state=seed)

        data_train = (
            X_train,
            y_train.squeeze(),
            w_train.squeeze(),
            mu0_train.squeeze(),
            mu1_train.squeeze(),
            cate_train.squeeze(),
            None,
        )
        data_val = (
            X_val,
            y_val.squeeze(),
            w_val.squeeze(),
            mu0_val.squeeze(),
            mu1_val.squeeze(),
            cate_val.squeeze(),
            None,
        )
        data_test = (
            X_test,
            y_test.squeeze(),
            w_test.squeeze(),
            mu_0_test.squeeze(),
            mu_1_test.squeeze(),
            cate_test.squeeze(),
            None,
        )

        return data_train, data_val, data_test
