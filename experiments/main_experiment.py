import csv
import os

import numpy as np
import pandas as pd
from sklearn import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

from catesel.meta_learners.base import SLearner, TLearner
from catesel.meta_learners.twostage_estimators import DRLearner, PWLearner, RALearner, RLearner
from catesel.model_selection.factual_scorers import FactualTEScorer, wFactualTEScorer
from catesel.model_selection.misc_scorers import IFTEScorer, MatchScorer
from catesel.model_selection.oracle_scorers import OraclePOScorer, OracleTEScorer
from catesel.model_selection.plugin_scorers import PluginTEScorer
from catesel.model_selection.pseudooutcome_scorers import PseudoOutcomeTEScorer, RTEScorer

RESULT_DIR = "results/"

LARGE_VAL = 999999999

# utils for pre-selecting hyperparameters (for computational time savings)
XGB_CV_NAME = "xgb_cv"
SEP = "-"
TRAIN_T_LEARNERS = [x + SEP + XGB_CV_NAME for x in ["t", "dr", "dr_pocv", "ra", "dr_oracle_p"]]
TRAIN_T_CRITS = [x + SEP + XGB_CV_NAME for x in ["dr_prefit", "ra_prefit", "if_prefit"]]
TRAIN_R_LEARNERS = [x + SEP + XGB_CV_NAME for x in ["r", "r_pocv"]]
TRAIN_R_CRITS = "r_prefit" + SEP + XGB_CV_NAME

VAL_T_MODS = [x + SEP + XGB_CV_NAME for x in ["t", "dr", "if", "dr_plug"]]
VAL_R_MODS = [x + SEP + XGB_CV_NAME for x in ["r", "r_plug"]]


XGB_PARAM_GRID = {"learning_rate": [0.1, 0.3], "max_depth": [1, 3, 6], "n_estimators": [20, 100]}


def _presel_hyperparams(X, y, w, n_cv=5):
    # (factually) select hyperparams for all po models -- to save computation costs
    # unconditional mean (for R learner)
    uncond_mean = GridSearchCV(XGBRegressor(verbosity=0), param_grid=XGB_PARAM_GRID, cv=n_cv)
    uncond_mean.fit(X, y)
    r_train_params = [uncond_mean.best_params_]

    # po0 & po1
    po0 = GridSearchCV(XGBRegressor(verbosity=0), param_grid=XGB_PARAM_GRID, cv=n_cv)
    po0.fit(X[w == 0, :], y[w == 0])
    po1 = GridSearchCV(XGBRegressor(verbosity=0), param_grid=XGB_PARAM_GRID, cv=n_cv)
    po1.fit(X[w == 1, :], y[w == 1])
    t_train_params = [po0.best_params_, po1.best_params_]

    return r_train_params, t_train_params


def run_one_experiment(
    input_data_train,
    input_data_val,
    input_data_test,
    models_to_compare,
    criteria_to_compare,
    oracle_baselines=None,
    pre_sel=True,
):
    # function to run a single experiment comparing multiple models given three sets of data
    X_train, y_train, w_train, mu0_train, mu1_train, cate_train, p_train = input_data_train

    # MODEL TRAINING -----
    # pre-processing step: do hyperparams selection for T-learner xgb-cv upfront so it can be
    # used by all learners (this does not lead to leakage and is the same as doing it within,
    # but saves some time for experiments)
    if pre_sel:
        r_train_params, t_train_params = _presel_hyperparams(X_train, y_train, w_train)

    # train models on training data
    for name, model in models_to_compare.items():
        print("Training model {}".format(name))

        if name in TRAIN_T_LEARNERS and pre_sel:
            model.set_params(est_params=t_train_params)
        elif name in TRAIN_R_LEARNERS and pre_sel:
            model.set_params(est_params=r_train_params)

        if "oracle_p" in name:
            model.fit(X_train, y_train, w_train, p=p_train)
        else:
            model.fit(X_train, y_train, w_train)

    if oracle_baselines is not None:
        print("Training oracles.")
        for oracle_name, oracle_model in oracle_baselines.items():
            oracle_model.fit(X_train, cate_train)

    # VALIDATION -----
    # use all validation criteria to find a 'best model' for each criterion

    # get data
    X_val, y_val, w_val, mu0_val, mu1_val, cate_val, p_val = input_data_val

    # pre-processing step: do hyperparams selection for T-learner xgb-cv upfront so it can be
    # used by all learners (no leakage, same as doing this within)
    if pre_sel:
        r_val_params, t_val_params = _presel_hyperparams(X_val, y_val, w_val)

    val_scores = []
    best_models = {}
    best_model_list = []
    # loop through criteria
    for crit_name, criterion in criteria_to_compare.items():
        criterion.reset_models()  # just in case

        if hasattr(criterion, "plugin_prefit"):
            if criterion.plugin_prefit:
                # prefit models on train data!
                if crit_name in TRAIN_T_CRITS and pre_sel:
                    criterion.set_est_params(t_train_params)
                elif crit_name in TRAIN_R_CRITS and pre_sel:
                    criterion.set_est_params(r_train_params)
                criterion.fit_plugin_model(X_train, y_train, w_train)

        if crit_name in VAL_T_MODS and pre_sel:
            criterion.set_est_params(t_val_params)
        elif crit_name in VAL_R_MODS and pre_sel:
            criterion.set_est_params(r_val_params)

        val_scores_crit = {}
        for model_name, model in models_to_compare.items():
            if "oracle_p" in crit_name:
                next_val = criterion(model, X_val, y_val, w_val, p_true=p_val, t_true=cate_val)

            elif crit_name == "po_oracle":
                next_val = criterion(model, X_val, y_val, w_val, p_true=None, t_true=[mu0_val, mu1_val])
            else:
                next_val = criterion(model, X_val, y_val, w_val, p_true=None, t_true=cate_val)
            val_scores_crit.update({model_name: next_val if (not np.isnan(next_val)) else LARGE_VAL})
        val_scores += list(val_scores_crit.values())
        # save name of best model
        print("Criterion: {}, scores: {}".format(crit_name, val_scores_crit))
        best_models.update({crit_name: min(val_scores_crit, key=val_scores_crit.get)})
        best_model_list.append(min(val_scores_crit, key=val_scores_crit.get))

    print("Best models: {}".format(best_models))

    # TEST --------------------------------
    # provide oracle score for each
    X_test, y_test, w_test, mu0_test, mu1_test, cate_test, p_test = input_data_test
    crit_names = []
    rmse_cate = []
    rmse_mu0 = []
    rmse_mu1 = []
    rmse_factual = []
    for crit_name, best_model_name in best_models.items():
        try:
            cate_hat, po0_hat, po1_hat = models_to_compare[best_model_name].predict(X_test, return_po=True)
            rmse_mu0.append(np.sqrt(mean_squared_error(po0_hat, mu0_test)))
            rmse_mu1.append(np.sqrt(mean_squared_error(po1_hat, mu1_test)))
            rmse_factual.append(np.sqrt(mean_squared_error(w_test * po1_hat + (1 - w_test) * po0_hat, y_test)))
        except ValueError:
            # pseudooutcome learners do not output CATE estimates
            cate_hat = models_to_compare[best_model_name].predict(X_test)
            rmse_mu0.append(np.nan)
            rmse_mu1.append(np.nan)
            rmse_factual.append(np.nan)

        rmse_cate.append(np.sqrt(mean_squared_error(cate_hat, cate_test)))

        crit_names.append(crit_name)  # for safety

    test_pehe_models = []
    test_factual_models = []
    test_pos_models = []
    for name, model in models_to_compare.items():
        try:
            cate_hat, po0_hat, po1_hat = model.predict(X_test, return_po=True)
            test_factual_models.append(np.sqrt(mean_squared_error(w_test * po1_hat + (1 - w_test) * po0_hat, y_test)))
            test_pos_models.append(
                mean_squared_error(po1_hat, mu1_test) / 2 + mean_squared_error(po0_hat, mu0_test) / 2
            )
        except ValueError:
            cate_hat = model.predict(X_test)
            test_factual_models.append(np.nan)
            test_pos_models.append(np.nan)

        test_pehe_models.append(np.sqrt(mean_squared_error(cate_hat, cate_test)))

    real_scores = []

    # also compute performance of criterion itself:
    for crit_name, crit_model in criteria_to_compare.items():
        if "oracle_p" in crit_name:
            real_scores.append(
                crit_model(None, X_test, y_test, w_test, p_true=p_test, t_true=cate_test, t_score=cate_test)
            )
        else:
            real_scores.append(
                crit_model(None, X_test, y_test, w_test, p_true=None, t_true=cate_test, t_score=cate_test)
            )

    oracle_scores = []
    if oracle_baselines is not None:
        # also include oracle baseline scores to see best achievable in class performance
        for oracle_name, oracle_model in oracle_baselines.items():
            oracle_scores.append(np.sqrt(mean_squared_error(cate_test, oracle_model.predict(X_test))))

    return (
        best_model_list
        + rmse_cate
        + real_scores
        + rmse_mu0
        + rmse_mu1
        + rmse_factual
        + val_scores
        + test_pehe_models
        + test_factual_models
        + test_pos_models
        + oracle_scores
    )


def run_repeat_experiment(
    models_to_compare,
    criteria_to_compare,
    dgp_to_sample,
    oracle_baselines=None,
    n_exp=10,
    file_name=None,
    crit_id=None,
    learner_id=None,
    return_frame=False,
):
    header = (
        ["setting_name", "seed", "criterion_id", "learner_id", "cate_var"]
        + ["best_" + x for x in criteria_to_compare.keys()]
        + ["rmse_cate_" + x for x in criteria_to_compare.keys()]
        + ["truth_score_" + x for x in criteria_to_compare.keys()]
        + ["rmse_mu0_" + x for x in criteria_to_compare.keys()]
        + ["rmse_mu1_" + x for x in criteria_to_compare.keys()]
        + ["rmse_factual_" + x for x in criteria_to_compare.keys()]
        + ["val_" + x + "_" + y for x in criteria_to_compare.keys() for y in models_to_compare.keys()]
        + ["test_pehe_" + x for x in models_to_compare.keys()]
        + ["test_factual_" + x for x in models_to_compare.keys()]
        + ["test_pos_" + x for x in models_to_compare.keys()]
        + (["test_pehe_" + x for x in oracle_baselines.keys()] if oracle_baselines is not None else [])
    )

    setting_name = dgp_to_sample.name

    out_frame = pd.DataFrame(columns=header)

    if file_name is not None:
        # if file_name does not exist yet
        if not os.path.exists(RESULT_DIR):
            os.makedirs(RESULT_DIR)

        if not os.path.isfile(RESULT_DIR + (file_name + ".csv")):
            out_file = open(RESULT_DIR + (file_name + ".csv"), "w", newline="", buffering=1)
            writer = csv.writer(out_file)
            writer.writerow(header)
        else:
            out_file = open(RESULT_DIR + (file_name + ".csv"), "a", newline="", buffering=1)
            writer = csv.writer(out_file)

    if isinstance(n_exp, int):
        n_exp = range(n_exp)

    for i in n_exp:
        data_train, data_val, data_test = dgp_to_sample(seed=i)
        # try:
        out_i = run_one_experiment(
            input_data_train=data_train,
            input_data_val=data_val,
            input_data_test=data_test,
            models_to_compare=models_to_compare,
            criteria_to_compare=criteria_to_compare,
            oracle_baselines=oracle_baselines,
        )
        # except:
        #  continue

        if file_name is not None:
            # append row to file if it exists
            writer.writerow([setting_name, i, crit_id, learner_id, np.var(data_train[5])] + out_i)

        if return_frame:
            new_frame = pd.DataFrame(
                columns=header, data=[[setting_name, i, crit_id, learner_id, np.var(data_train[5])] + out_i]
            )
            out_frame = pd.concat([out_frame, new_frame])

    out_file.close()
    if return_frame:
        return out_frame


def get_model(model_name, hyperparams=None, n_cv=5):
    if hyperparams is None:
        hyperparams = {}

    if model_name == "lr":
        alphas = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100, 1000, 10000]
        regressor = RidgeCV(alphas=alphas, **hyperparams)
    elif model_name == "rf":
        hyperparam_dict = {"max_depth": 4, "n_estimators": 400}  # min samples leaf = 5
        hyperparams.update({key: value for key, value in hyperparam_dict.items() if key not in hyperparams.keys()})
        regressor = RandomForestRegressor(**hyperparams)
    elif model_name == "rf_cv":
        regressor = GridSearchCV(
            RandomForestRegressor(),
            param_grid={
                "max_depth": [2, 3, 4, None],
                "n_estimators": [200]
                # "min_samples_leaf": [2, 3, 5, 10],
                # 'n_estimators': [100, 200, 500]
            },
            cv=n_cv,
        )
    elif model_name == "gb":
        regressor = GradientBoostingRegressor(n_estimators=100, **hyperparams)
    elif model_name == "xgb":
        regressor = XGBRegressor(learning_rate=0.1, max_depth=3, verbosity=0)
    elif model_name == "xgb_cv":
        regressor = GridSearchCV(XGBRegressor(verbosity=0), param_grid=XGB_PARAM_GRID, cv=n_cv)
    else:
        ValueError("Invalid regressor name")

    return regressor


def get_learner(learner_name, learner_regressor, prop_model=None, n_folds=5):
    inner_cv = isinstance(learner_regressor, GridSearchCV)
    if inner_cv:
        param_grid = learner_regressor.param_grid
    else:
        param_grid = None

    inner_regressor = learner_regressor if (not inner_cv) else learner_regressor.estimator
    learner_regressor = clone(learner_regressor)
    inner_regressor = clone(inner_regressor)
    learner_dict = {
        "s": SLearner(po_estimator=learner_regressor),
        "s_ext": SLearner(po_estimator=learner_regressor, extend_covs=True),
        "t": TLearner(po_estimator=learner_regressor),
        "dr": DRLearner(
            po_estimator=learner_regressor,
            n_folds=n_folds,
            te_estimator=learner_regressor,
            propensity_estimator=LogisticRegressionCV(Cs=[0.00001, 0.001, 0.01, 0.1, 1])
            if prop_model is None
            else prop_model,
        ),
        "dr_pocv": DRLearner(
            po_estimator=inner_regressor,
            te_estimator=inner_regressor,
            avg_fold_models=True,
            propensity_estimator=LogisticRegressionCV(Cs=[0.00001, 0.001, 0.01, 0.1, 1])
            if prop_model is None
            else prop_model,
            n_folds=n_folds,
            pre_cv_po=inner_cv,
            pre_cv_te=inner_cv,
            grid_po=param_grid,
        ),
        "dr_oracle_p": DRLearner(
            po_estimator=learner_regressor,
            n_folds=n_folds,
            te_estimator=learner_regressor,
            fit_propensity_estimator=False,
        ),
        "ra": RALearner(
            po_estimator=learner_regressor, n_folds=1, te_estimator=learner_regressor, fit_propensity_estimator=False
        ),
        "r": RLearner(
            po_estimator=learner_regressor,
            n_folds=n_folds,
            te_estimator=learner_regressor,
            propensity_estimator=LogisticRegressionCV(Cs=[0.00001, 0.001, 0.01, 0.1, 1])
            if prop_model is None
            else prop_model,
        ),
        "r_pocv": RLearner(
            po_estimator=inner_regressor,
            te_estimator=inner_regressor,
            avg_fold_models=True,
            propensity_estimator=LogisticRegressionCV(Cs=[0.00001, 0.001, 0.01, 0.1, 1])
            if prop_model is None
            else prop_model,
            n_folds=n_folds,
            pre_cv_po=inner_cv,
            pre_cv_te=inner_cv,
            grid_po=param_grid,
        ),
        "r_oracle_p": RLearner(
            po_estimator=learner_regressor,
            n_folds=n_folds,
            te_estimator=learner_regressor,
            fit_propensity_estimator=False,
        ),
        "pw": PWLearner(
            po_estimator=learner_regressor,
            n_folds=1,
            te_estimator=learner_regressor,
            fit_propensity_estimator=True,
            propensity_estimator=LogisticRegressionCV(Cs=[0.00001, 0.001, 0.01, 0.1, 1])
            if prop_model is None
            else prop_model,
        ),
    }
    return learner_dict[learner_name]


def get_criterion(criterion_name, criterion_regressor, prop_model=None):
    inner_cv = isinstance(criterion_regressor, GridSearchCV)
    if inner_cv:
        param_grid = criterion_regressor.param_grid
    else:
        param_grid = None

    inner_regressor = criterion_regressor if (not inner_cv) else criterion_regressor.estimator
    criterion_regressor = clone(criterion_regressor) if criterion_regressor is not None else criterion_regressor
    inner_regressor = clone(inner_regressor) if inner_regressor is not None else inner_regressor
    lr_Cs = [0.00001, 0.001, 0.01, 0.1, 1]
    crit_dict = {
        "factual": FactualTEScorer(mean_squared_error, 1),
        "w_factual": wFactualTEScorer(
            mean_squared_error, 1, prop_model=LogisticRegressionCV(Cs=lr_Cs) if prop_model is None else prop_model
        ),
        "w_factual_prefit": wFactualTEScorer(
            mean_squared_error,
            1,
            prop_model=LogisticRegressionCV(Cs=lr_Cs) if prop_model is None else prop_model,
            plugin_prefit=True,
        ),
        "oracle": OracleTEScorer(mean_squared_error, 1),
        "po_oracle": OraclePOScorer(mean_squared_error, 1),
        "s": PluginTEScorer(mean_squared_error, 1, po_model=SLearner(po_estimator=criterion_regressor)),
        "t": PluginTEScorer(mean_squared_error, 1, po_model=TLearner(po_estimator=criterion_regressor)),
        "s_ext": PluginTEScorer(
            mean_squared_error, 1, po_model=SLearner(po_estimator=criterion_regressor, extend_covs=True)
        ),
        "dr_plug": PluginTEScorer(
            mean_squared_error,
            1,
            po_model=DRLearner(
                te_estimator=criterion_regressor,
                po_estimator=criterion_regressor,
                propensity_estimator=LogisticRegressionCV(Cs=[0.00001, 0.001, 0.01, 0.1, 1])
                if prop_model is None
                else prop_model,
                n_folds=1,
            ),
        ),
        "r_plug": PluginTEScorer(
            mean_squared_error,
            1,
            po_model=RLearner(
                te_estimator=criterion_regressor,
                po_estimator=criterion_regressor,
                propensity_estimator=LogisticRegressionCV(Cs=[0.00001, 0.001, 0.01, 0.1, 1])
                if prop_model is None
                else prop_model,
                n_folds=1,
            ),
        ),
        "ra": PseudoOutcomeTEScorer(
            mean_squared_error, 1, po_model=TLearner(po_estimator=criterion_regressor), pseudo_type="RA", n_folds=1
        ),
        "pw": PseudoOutcomeTEScorer(
            mean_squared_error,
            1,
            po_model=TLearner(po_estimator=criterion_regressor),
            prop_model=LogisticRegressionCV(Cs=lr_Cs) if prop_model is None else prop_model,
            pseudo_type="PW",
            n_folds=1,
        ),
        "dr": PseudoOutcomeTEScorer(
            mean_squared_error,
            1,
            po_model=TLearner(po_estimator=criterion_regressor) if not inner_cv else inner_regressor,
            prop_model=LogisticRegressionCV(Cs=lr_Cs) if prop_model is None else prop_model,
            n_folds=5,
            pseudo_type="DR",
            pre_cv_po=inner_cv,
            grid_po=param_grid,
        ),
        "dr_oracle_p": PseudoOutcomeTEScorer(
            mean_squared_error,
            1,
            po_model=TLearner(po_estimator=criterion_regressor),
            fit_propensity_estimator=False,
            n_folds=5,
            pseudo_type="DR",
        ),
        "r": RTEScorer(
            None,
            None,
            po_model=criterion_regressor if not inner_cv else inner_regressor,
            prop_model=LogisticRegressionCV(Cs=lr_Cs) if prop_model is None else prop_model,
            pseudo_type="R",
            n_folds=5,
            pre_cv_po=inner_cv,
            grid_po=param_grid,
        ),
        "r_oracle_p": RTEScorer(
            None, None, po_model=criterion_regressor, fit_propensity_estimator=False, pseudo_type="R", n_folds=5
        ),
        "if": IFTEScorer(
            None,
            None,
            plugin_prefit=False,
            po_model=TLearner(po_estimator=criterion_regressor) if not inner_cv else inner_regressor,
            prop_model=LogisticRegressionCV(Cs=lr_Cs) if prop_model is None else prop_model,
            n_folds=5,
            pre_cv_po=inner_cv,
            grid_po=param_grid,
        ),
        "if_prefit": IFTEScorer(
            None,
            None,
            plugin_prefit=False,
            po_model=TLearner(po_estimator=criterion_regressor),
            prop_model=LogisticRegressionCV(Cs=lr_Cs) if prop_model is None else prop_model,
        ),
        "dr_prefit": PseudoOutcomeTEScorer(
            mean_squared_error,
            1,
            plugin_prefit=True,
            po_model=TLearner(po_estimator=criterion_regressor),
            prop_model=LogisticRegressionCV(Cs=lr_Cs) if prop_model is None else prop_model,
            pseudo_type="DR",
        ),
        "dr_prefit_oracle_p": PseudoOutcomeTEScorer(
            mean_squared_error,
            1,
            plugin_prefit=True,
            po_model=TLearner(po_estimator=criterion_regressor),
            fit_propensity_estimator=False,
            pseudo_type="DR",
        ),
        "ra_prefit": PseudoOutcomeTEScorer(
            mean_squared_error,
            1,
            plugin_prefit=True,
            po_model=TLearner(po_estimator=criterion_regressor),
            pseudo_type="RA",
        ),
        "r_prefit": RTEScorer(
            None,
            None,
            plugin_prefit=True,
            po_model=criterion_regressor,
            prop_model=LogisticRegressionCV(Cs=lr_Cs) if prop_model is None else prop_model,
            pseudo_type="R",
        ),
        "r_prefit_oracle_p": RTEScorer(
            None,
            None,
            plugin_prefit=True,
            po_model=criterion_regressor,
            fit_propensity_estimator=False,
            pseudo_type="R",
        ),
        "match": MatchScorer(mean_squared_error, 1),
    }
    return crit_dict[criterion_name]
