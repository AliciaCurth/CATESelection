import glob
import os
from pathlib import Path

import gdown
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data_utils import download_if_needed
from .experiment_utils import DataGenerator

DATA_DIR = "data/acic/"
ACIC_COV_TRANS = "x_trans.csv"
ACIC_COV = "x.csv"
ACIC_ORIG_DIR = "data_cf_all/"
RESULT_DIR_SIMU = "results/acic_simu/"
PREPROCESSED_FILE_ID = "1iOfEAk402o3jYBs2Prfiz6oaailwWcR5"
ACIC_ORIG_ID = "0B7pG5PPgj6A3N09ibmFwNWE1djA"
SEP = "_"

NUMERIC_COLS = [0, 3, 4, 16, 17, 18, 20, 21, 22, 23, 24, 24, 25, 30, 31, 32, 33, 39, 40, 41, 53, 54]
N_NUM_COLS = len(NUMERIC_COLS)


def download_orig_acic():
    data_path = Path(DATA_DIR)
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    download_if_needed(
        data_path / "data_cf_all.tar.gz", file_id=ACIC_ORIG_ID, unarchive=True, unarchive_folder=data_path
    )


def download_input_trans():
    cov_loc = DATA_DIR + ACIC_COV_TRANS
    if not Path(cov_loc).exists():
        data_path = Path(DATA_DIR)  # noqa: F841 # pylint: disable=unused-variable
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)

        gdown.download(id=PREPROCESSED_FILE_ID, output=cov_loc, quiet=False)


def randomly_binarize_numeric_cols(X, numeric_only=False, numeric_cols=tuple(NUMERIC_COLS)):
    X_new = X.copy()
    for i in numeric_cols:
        X_new[:, i] = (X[:, i] > np.random.choice(X[:, i])).astype(int)
    return X_new[:, numeric_cols] if numeric_only else X_new


def get_acic_covariates(pre_trans: bool = False, keep_categorical: bool = False):
    if pre_trans:
        download_input_trans()
        X_t = pd.read_csv(DATA_DIR + ACIC_COV_TRANS)
        if not keep_categorical:
            X_t = X_t.drop(columns=["x_2", "x_21", "x_24"])
        X_t = X_t.values
    else:
        download_orig_acic()
        X = pd.read_csv(DATA_DIR + ACIC_ORIG_DIR + ACIC_COV)
        if not keep_categorical:
            X = X.drop(columns=["x_2", "x_21", "x_24"])
        else:
            # encode categorical features
            feature_list = []
            for cols_ in X.columns:
                if type(X.loc[X.index[0], cols_]) not in [np.int64, np.float64]:
                    enc = OneHotEncoder(drop="first")

                    enc.fit(np.array(X[[cols_]]).reshape((-1, 1)))

                    for k in range(len(list(enc.get_feature_names()))):  # pyright: ignore
                        X[cols_ + list(enc.get_feature_names())[k]] = enc.transform(  # pyright: ignore
                            np.array(X[[cols_]]).reshape((-1, 1))
                        ).toarray()[  # pyright: ignore
                            :, k
                        ]

                    feature_list.append(cols_)

            X.drop(feature_list, axis=1, inplace=True)

        scaler = StandardScaler()
        X_t = scaler.fit_transform(X)
    return X_t


def get_filenames(simu_num):
    return sorted(glob.glob(DATA_DIR + ACIC_ORIG_DIR + str(simu_num) + "/zymu_*.csv"))


def get_outcomes(filename):
    out = pd.read_csv(filename)
    w = out["z"]
    y = w * out["y1"] + (1 - w) * out["y0"]
    mu0 = out["mu0"]
    mu1 = out["mu1"]
    cate_true = out["mu1"] - out["mu0"]
    return y.values, w.values, mu0.values, mu1.values, cate_true.values


def get_acic_data_orig(simu_num, seed, n_test, subset_train=None, pre_trans=True, subset_cols="all"):
    X = get_acic_covariates(pre_trans=pre_trans)
    # remove binary cols if wanted
    if subset_cols == "numeric" and pre_trans:
        X = X[:, NUMERIC_COLS]
    elif subset_cols == "other" and pre_trans:
        X = X[:, [x for x in range(X.shape[1]) if x not in NUMERIC_COLS]]
    elif subset_cols == "all":
        pass
    else:
        raise ValueError("Invalid column type")

    download_orig_acic()

    file_list = get_filenames(simu_num)

    # get data
    y_full, w_full, mu0_full, mu1_full, cate_full = get_outcomes(file_list[seed])

    # split data
    X, X_t, y, y_t, w, w_t, mu0, mu0_t, mu1, mu1_t, cate_in, cate_out = split_data(
        X,
        y_full,
        w_full,
        mu0_full,
        mu1_full,
        cate_full,
        n_test=n_test,
        random_state=seed,
        subset_train=subset_train,  # pyright: ignore
    )
    return X, X_t, y, y_t, w, w_t, mu0, mu0_t, mu1, mu1_t, cate_in, cate_out


def split_data(
    X_full,
    y_full,
    w_full,
    mu0_full,
    mu1_full,
    cate_full,
    n_test=0.8,
    random_state=42,
    subset_train: int = None,  # type: ignore
):
    X, X_t, y, y_t, w, w_t, mu0, mu0_t, mu1, mu1_t, cate_in, cate_out = train_test_split(
        X_full, y_full, w_full, mu0_full, mu1_full, cate_full, test_size=n_test, random_state=random_state
    )

    if subset_train is not None:
        X, y, w, mu0, mu1, cate_in = (
            X[:subset_train, :],
            y[:subset_train],
            w[:subset_train],
            mu0[:subset_train],
            mu1[:subset_train],
            cate_in[:subset_train],
        )

    return X, X_t, y, y_t, w, w_t, mu0, mu0_t, mu1, mu1_t, cate_in, cate_out


def acic_simu(
    i_exp,
    n_train=1000,
    n_test=500,
    error_sd: float = 1,
    sp_lin: float = 0.6,
    sp_nonlin: float = 0.3,
    sp_1: float = 0,
    p_0: float = 0,
    ate_goal: float = 0,
    inter: int = 3,
    inter_t: int = 0,
    return_ytest: bool = False,
    col_type="all",
    xi=0.5,
    propensity_type="random",
    rel_prop=0.5,
    pre_trans=True,
    nonlin_scale=0.1,
    r_spec=False,
    prop_truespec=True,
):
    np.random.seed(i_exp)

    # get data
    X_full = get_acic_covariates(pre_trans=True)
    _, ncov_full = X_full.shape

    # remove binary cols if wanted
    if col_type == "numeric":
        X = X_full[:, NUMERIC_COLS]
    elif col_type == "other":
        X = X_full[:, [x for x in range(X_full.shape[1]) if x not in NUMERIC_COLS]]
    elif col_type == "all":
        X = X_full
    elif col_type == "binarized":
        X = randomly_binarize_numeric_cols(X_full)
    elif col_type == "missp_out":
        X = randomly_binarize_numeric_cols(X_full, numeric_only=True)
    else:
        raise ValueError("Invalid column type")

    cov_ratio = ncov_full / X.shape[1]

    # return untransformed data if required
    if pre_trans == 0:
        X_return = X_full  # no transformation at all
    elif pre_trans == 1:  # transform data but do not give away which columns are used
        X_return = X_full.copy()
        if col_type == "binarized":
            X_return = X
        elif col_type == "missp_out":
            X_return[:, NUMERIC_COLS] = X.copy()
    else:
        X_return = X

    # shuffle indices
    n_total, n_cov = X.shape
    ind = np.arange(n_total)
    np.random.shuffle(ind)
    ind_test = ind[-n_test:]

    # create dgp
    coeffs_ = [0, 1]
    beta_0 = np.random.choice(coeffs_, size=n_cov, replace=True, p=[1 - sp_lin * cov_ratio, sp_lin * cov_ratio])
    intercept = np.random.choice([x for x in np.arange(-1, 1.25, 0.25)])
    beta_1 = np.random.choice(coeffs_, size=n_cov, replace=True, p=[1 - sp_1 * cov_ratio, sp_1 * cov_ratio])
    mu_1_mask = np.random.choice([0, 1], replace=True, size=n_cov, p=[p_0, 1 - p_0])

    # simulate mu_0 and mu_1
    if not r_spec:
        mu_0 = (intercept + np.dot(X, beta_0)).reshape((-1, 1))
        mu_1 = (intercept + np.dot(X, beta_1 + beta_0 * mu_1_mask)).reshape((-1, 1))
    else:
        mu_0 = (intercept + np.dot(X, beta_0) - 0.5 * np.dot(X, beta_1)).reshape((-1, 1))
        mu_1 = (intercept + np.dot(X, 0.5 * beta_1 + beta_0 * mu_1_mask)).reshape((-1, 1))
    coefs_sq = [0, nonlin_scale]
    if sp_nonlin > 0:
        if ((not col_type == "other") and (not col_type == "binarized")) and (not col_type == "missp_out"):
            beta_sq = np.random.choice(
                coefs_sq, size=N_NUM_COLS, replace=True, p=[1 - sp_nonlin * cov_ratio, sp_nonlin * cov_ratio]
            )
            mu_1_mask_sq = np.random.choice([0, 1], replace=True, size=N_NUM_COLS, p=[p_0, 1 - p_0])
            X_sq = X[:, NUMERIC_COLS] ** 2 if col_type == "all" else X

            mu_0 = mu_0 + np.dot(X_sq, beta_sq).reshape((-1, 1))
            mu_1 = mu_1 + np.dot(X_sq, beta_sq * mu_1_mask_sq).reshape((-1, 1))

        if inter > 0:
            # randomly add some interactions
            ind_c = np.arange(n_cov)
            np.random.shuffle(ind_c)
            inter_list = list()
            for i in range(0, n_cov - 2, 2):
                inter_list.append(X[:, ind_c[i]] * X[:, ind_c[i + 1]])

            if inter > 1:
                np.random.shuffle(ind_c)
                for i in range(0, n_cov - 3, 3):
                    inter_list.append(X[:, ind_c[i]] * X[:, ind_c[i + 1]] * X[:, ind_c[i + 2]])

            if inter > 2:
                np.random.shuffle(ind_c)
                for i in range(0, n_cov - 4, 4):
                    inter_list.append(X[:, ind_c[i]] * X[:, ind_c[i + 1]] * X[:, ind_c[i + 2]] * X[:, ind_c[i + 3]])

            X_inter = np.array(inter_list).T
            n_inter = X_inter.shape[1]
            beta_inter = np.random.choice(
                coefs_sq, size=n_inter, replace=True, p=[1 - sp_nonlin * cov_ratio, sp_nonlin * cov_ratio]
            )
            mu_1_mask_inter = np.random.choice([0, 1], replace=True, size=n_inter, p=[p_0, 1 - p_0])
            mu_0 = mu_0 + np.dot(X_inter, beta_inter).reshape((-1, 1))
            mu_1 = mu_1 + np.dot(X_inter, beta_inter * mu_1_mask_inter).reshape((-1, 1))

    if inter_t > 0:
        ind_c = np.arange(n_cov)
        np.random.shuffle(ind_c)
        inter_list_t = list()
        for i in range(0, n_cov - 2, 2):
            inter_list_t.append(X[:, ind_c[i]] * X[:, ind_c[i + 1]])

        if inter_t > 1:
            np.random.shuffle(ind_c)
            for i in range(0, n_cov - 3, 3):
                inter_list_t.append(X[:, ind_c[i]] * X[:, ind_c[i + 1]] * X[:, ind_c[i + 2]])

        if inter_t > 2:
            np.random.shuffle(ind_c)
            for i in range(0, n_cov - 4, 4):
                inter_list_t.append(X[:, ind_c[i]] * X[:, ind_c[i + 1]] * X[:, ind_c[i + 2]] * X[:, ind_c[i + 3]])

        X_inter_t = np.array(inter_list_t).T
        n_inter = X_inter_t.shape[1]
        beta_inter_t = np.random.choice(
            coefs_sq, size=n_inter, replace=True, p=[1 - sp_1 * cov_ratio, sp_1 * cov_ratio]
        )

        if not r_spec:
            mu_1 = mu_1 + np.dot(X_inter_t, beta_inter_t).reshape((-1, 1))
        else:
            mu_0 = mu_0 - 0.5 * np.dot(X_inter_t, beta_inter_t).reshape((-1, 1))
            mu_1 = mu_1 + 0.5 * np.dot(X_inter_t, beta_inter_t).reshape((-1, 1))

    ate = np.mean(mu_1 - mu_0)
    mu_1 = mu_1 - ate + ate_goal

    if prop_truespec:
        # propensity not misspecified: should use what the learners have available
        if col_type == "missp_out" and not pre_trans:
            # use the numeric cols
            X_prop = X_full[:, NUMERIC_COLS]
        else:
            # use exactly what is used for prediction
            X_prop = X
    else:
        # propensity misspecified: should use the transformation the learners do not have
        if col_type == "missp_out" and pre_trans:
            # learners get x so use opposite
            X_prop = X_full[:, NUMERIC_COLS]
        else:
            # learners do not have pre transformed x so should use it here
            X_prop = X

    if propensity_type == "random":
        p = xi * np.ones(n_total)

    elif propensity_type == "irrelevant_var":
        # sample using propensity with unrelated variables
        coeff = np.ones(n_cov) * (beta_0 + beta_1 == 0)
        exponent = zscore(np.dot(X_prop, coeff)) if np.sum(coeff) > 0 else np.zeros_like(np.dot(X_prop, coeff))
        p = expit(xi * exponent)
    elif propensity_type == "pred":
        # sample using propensity based on predictive variables
        exponent = zscore(np.dot(X_prop, beta_1 + (1 - mu_1_mask)))
        p = expit(xi * exponent)
    elif propensity_type == "prog":
        # sample using prognostic variables
        exponent = zscore(np.dot(X_prop, beta_0 * mu_1_mask))
        p = expit(xi * exponent)
    else:
        raise ValueError("Invalid propensity_type")

    # create treatment indicator (treatment assignment does not matter in test set)
    w = np.random.binomial(1, p=p * (rel_prop / 0.5)).reshape(-1, 1)

    y = w * mu_1 + (1 - w) * mu_0 + np.random.normal(0, error_sd, n_total).reshape((-1, 1))

    cate = mu_1 - mu_0

    X_train, y_train, w_train, cate_train, mu0_train, mu1_train, p_train = (
        X_return[ind[:(n_train)], :],
        y[ind[:(n_train)]],
        w[ind[:(n_train)]],
        cate[ind[:(n_train)]],
        mu_0[ind[:(n_train)]],
        mu_1[ind[:(n_train)]],
        p[ind[:(n_train)]],
    )

    X_test, mu_0_t, mu_1_t, cate_t, p_t = (
        X_return[ind_test, :],
        mu_0[ind_test],
        mu_1[ind_test],
        cate[ind_test],
        p[ind_test],
    )

    if return_ytest:
        y_test, w_test = y[ind_test], w[ind_test]
        return (
            X_train,
            y_train,
            w_train,
            mu0_train,
            mu1_train,
            cate_train,
            p_train,
            X_test,
            y_test,
            w_test,
            mu_0_t,
            mu_1_t,
            cate_t,
            p_t,
        )

    return X_train, y_train, w_train, mu0_train, mu1_train, cate_train, p_train, X_test, mu_0_t, mu_1_t, cate_t, p_t


class AcicOrigGenerator(DataGenerator):
    # pylint: disable-next=super-init-not-called
    def __init__(self, n_train=1000, n_val=500, n_test=500, subset_cols="all", simu_num=1):
        self.n_train = n_train
        self.n_test = n_test
        self.n_val = n_val
        self.subset_cols = subset_cols
        self.simu_num = simu_num

        self.name = (
            "acicorig-" + str(simu_num) + "-" + str(n_train) + "-" + str(n_val) + "-" + str(n_test) + "-" + subset_cols
        )

    def __call__(self, seed=42):
        X, X_t, y, y_t, w, w_t, mu0, mu0_t, mu1, mu1_t, cate_in, cate_out = get_acic_data_orig(
            simu_num=self.simu_num,
            n_test=self.n_test,
            subset_train=self.n_train + self.n_val,
            subset_cols=self.subset_cols,
            seed=seed,
        )
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
        ) = train_test_split(
            X, w, y, mu0, mu1, cate_in, train_size=self.n_train, random_state=seed  # TODO
        )  # stratify on w

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
        data_test = X_t, y_t.squeeze(), w_t.squeeze(), mu0_t.squeeze(), mu1_t.squeeze(), cate_out.squeeze(), None

        return data_train, data_val, data_test


class AcicLinearGenerator(DataGenerator):
    # pylint: disable-next=super-init-not-called
    def __init__(
        self,
        n_train=1000,
        n_val=1000,
        n_test=500,
        sp_lin=0.6,
        sp_nonlin=0.3,
        sp_1=0,
        inter=True,
        xi=0.5,
        p_0=0,
        error_sd=1,
        ate_goal=0,
        subset_cols="all",
        propensity_type="random",
        trans_cov=True,
        inter_t=0,
        nonlin_scale=0.1,
        rel_prop=0.5,
        r_spec=False,
        prop_truespec=True,
    ):
        self.n_train = n_train
        self.n_val = n_val
        self.n_test = n_test
        self.sp_lin = sp_lin
        self.sp_nonlin = sp_nonlin
        self.sp_1 = sp_1
        self.p_0 = p_0
        self.inter = inter
        self.xi = xi
        self.error_sd = error_sd
        self.ate_goal = ate_goal
        self.subset_cols = subset_cols
        self.propensity_type = propensity_type
        self.trans_cov = trans_cov
        self.inter_t = inter_t
        self.nonlin_scale = nonlin_scale
        self.rel_prop = rel_prop
        self.r_spec = r_spec
        self.prop_truespec = prop_truespec

        self.name = (
            "aciclinnobias"
            + ("rspec" if r_spec else "")
            + ("propfalsespec" if not prop_truespec else "")
            + "-"
            + str(n_train)
            + "-"
            + str(n_val)
            + "-"
            + str(n_test)
            + "-"
            + str(sp_lin)
            + "-"
            + str(sp_nonlin)
            + "-"
            + str(sp_1)
            + "-"
            + str(p_0)
            + "-"
            + str(inter)
            + "-"
            + str(inter_t)
            + "-"
            + str(nonlin_scale)
            + "-"
            + propensity_type
            + "-"
            + str(xi)
            + "-"
            + str(rel_prop)
            + "-"
            + subset_cols
            + "-"
            + str(trans_cov)
            + "-"
            + str(error_sd)
        )

    def __call__(self, seed=42):
        (  # pyright: ignore
            X,
            y,
            w,
            mu0,
            mu1,
            cate,
            p,
            X_test,
            y_test,
            w_test,
            mu_0_test,
            mu_1_test,
            cate_test,
            p_test,
        ) = acic_simu(
            seed,
            n_train=self.n_train + self.n_val,
            xi=self.xi,
            propensity_type=self.propensity_type,
            n_test=self.n_test,
            error_sd=self.error_sd,
            sp_lin=self.sp_lin,
            sp_nonlin=self.sp_nonlin,
            sp_1=self.sp_1,
            ate_goal=self.ate_goal,
            inter=self.inter,
            p_0=self.p_0,
            return_ytest=True,
            col_type=self.subset_cols,
            prop_truespec=self.prop_truespec,
            pre_trans=self.trans_cov,
            inter_t=self.inter_t,
            nonlin_scale=self.nonlin_scale,
            rel_prop=self.rel_prop,
            r_spec=self.r_spec,
        )
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
            p_train,
            p_val,
        ) = train_test_split(
            X, w, y, mu0, mu1, cate, p, train_size=self.n_train, random_state=seed
        )  # stratify on w

        data_train = (
            X_train,
            y_train.squeeze(),
            w_train.squeeze(),
            mu0_train.squeeze(),
            mu1_train.squeeze(),
            cate_train.squeeze(),
            p_train.squeeze(),
        )
        data_val = (
            X_val,
            y_val.squeeze(),
            w_val.squeeze(),
            mu0_val.squeeze(),
            mu1_val.squeeze(),
            cate_val.squeeze(),
            p_val.squeeze(),
        )
        data_test = (
            X_test,
            y_test.squeeze(),
            w_test.squeeze(),
            mu_0_test.squeeze(),
            mu_1_test.squeeze(),
            cate_test.squeeze(),
            p_test.squeeze(),
        )

        return data_train, data_val, data_test
