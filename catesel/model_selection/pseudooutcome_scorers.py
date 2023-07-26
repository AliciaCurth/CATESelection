""""
Pseudo-outcome scorers, using a pseudo-outcome as surrogate for CATE
"""
# Author: Alicia Curth
import numpy as np
from sklearn import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from catesel.meta_learners.base import BasePluginCATEEstimator
from catesel.meta_learners.transformations import pseudo_outcome_transformation
from catesel.model_selection.base import _BaseTEScorer
from catesel.utils.base import _get_values_only


class PseudoOutcomeTEScorer(_BaseTEScorer):
    # CATE estimator scorer based on pseudo outcomes for CATE estimation: imputes pseudo
    # outcomes on validation data (or uses prefit model) and scores against that
    def __init__(
        self,
        score_func,
        sign,
        po_model=None,
        prop_model=None,
        plugin_prefit=False,
        pseudo_type="DR",
        n_folds=3,
        binary_y=False,
        random_state: int = 42,
        fit_propensity_estimator: bool = True,
        pre_cv_po=False,
        grid_po=None,
    ):
        self.po_model = po_model
        self.plugin_prefit = plugin_prefit
        self.prop_model = prop_model
        self.pseudo_type = pseudo_type
        self.n_folds = n_folds
        self.binary_y = binary_y
        self.random_state = random_state
        self.fit_propensity_estimator = fit_propensity_estimator
        self.pre_cv_po = pre_cv_po
        self.grid_po = grid_po

        self.est_params = None

        # initialise storage for models
        self._models_fitted = False
        self._fold_models_po = [None] * self.n_folds
        self._fold_pred_masks = [None] * self.n_folds
        self._fold_models_prop = [None] * self.n_folds
        super().__init__(score_func, sign)

    def reset_models(self):
        # reset any fold models
        self._models_fitted = False
        self._fold_models_po = [None] * self.n_folds
        self._fold_pred_masks = [None] * self.n_folds
        self._fold_models_prop = [None] * self.n_folds

        # reset pre-specified models
        self.po_model = clone(self.po_model) if self.po_model is not None else self.po_model
        self.prop_model = clone(self.prop_model) if self.prop_model is not None else self.prop_model

        self.po_params = []

    def set_est_params(self, est_params):
        if est_params is not None:
            if isinstance(est_params, list):
                self.est_params = est_params
            elif isinstance(est_params, dict):
                self.est_params = [est_params]
            else:
                raise ValueError("est_params should be a list of dicts or a dict.")

    def _do_po_cv(self, X, y, w):
        self.po_params = []
        temp_model_0 = GridSearchCV(self.po_model, param_grid=self.grid_po, cv=self.n_folds)
        temp_model_0.fit(X[w == 0], y[w == 0])
        self.po_params.append(temp_model_0.best_params_)

        # treated model
        temp_model_1 = GridSearchCV(self.po_model, param_grid=self.grid_po, cv=self.n_folds)
        temp_model_1.fit(X[w == 1], y[w == 1])
        self.po_params.append(temp_model_1.best_params_)

    def fit_plugin_model(self, X, y_factual, w_factual):
        if not self.plugin_prefit:
            return
        else:
            if self.est_params is not None:
                if isinstance(self.po_model, GridSearchCV):
                    self.po_model = clone(self.po_model.estimator)
                self.po_model.set_params(est_params=self.est_params)
            self.po_model.fit(X, y_factual, w_factual)
            if self.prop_model is not None:
                self.prop_model.fit(X, w_factual)
            self._models_fitted = True

    def _score(self, estimator, X, y_factual, w_factual, p_true=None, t_true=None, sample_weight=None, t_score=None):
        t_pred = estimator.predict(X, return_po=False) if t_score is None else t_score

        if not self.plugin_prefit and not self._models_fitted:
            # fit nuisance models
            mu0, mu1, prop = self._fit_and_impute_nuisance_components(X, y_factual, w_factual, p=p_true)
            self._models_fitted = True
        elif not self.plugin_prefit and self._models_fitted:
            # use nuisance models from storage
            mu0, mu1, prop = self._impute_nuisance_components(X, p=p_true)
        else:
            # use prefit model
            _, mu0, mu1 = self.po_model.predict(X, return_po=True)
            if p_true is not None:
                prop = p_true
            else:
                if self.prop_model is not None:
                    prop = self.prop_model.predict_proba(X)[:, 1]
                else:
                    prop = np.nan

        t_pseudo = pseudo_outcome_transformation(y_factual, w_factual, prop, mu0, mu1, pseudo_type=self.pseudo_type)

        if sample_weight is not None:
            return self._sign * self._score_func(
                t_pseudo,
                t_pred,
                sample_weight=sample_weight,
            )
        else:
            return self._sign * self._score_func(t_pseudo, t_pred)

    def _impute_nuisance_components(self, X, p=None):
        # impute the nuisance components from saved models
        if not self._models_fitted:
            raise ValueError("Internal models are not fitted")

        n, _ = X.shape
        mu_0_pred, mu_1_pred, p_pred = np.zeros(n), np.zeros(n), np.zeros(n)

        # if new data -- this usually does not happen, and is an artifact for benchmarking
        # experiments
        if not n == len(self._fold_pred_masks[0]):
            # just use first model for now
            pred_mask = np.ones(n, dtype=bool)
            if not self.pseudo_type == "PW":
                mu_0_pred[pred_mask], mu_1_pred[pred_mask] = self._impute_outcomes(X, None, None, None, pred_mask, 0)
            if self.pseudo_type == "DR" or self.pseudo_type == "PW" or self.pseudo_type == "R":
                p_pred[pred_mask] = self._impute_propensity(X, None, None, pred_mask, 0)
            else:
                p_pred[pred_mask] = np.nan

        else:
            for idx in range(self.n_folds):
                pred_mask = self._fold_pred_masks[idx]
                if not self.pseudo_type == "PW":
                    mu_0_pred[pred_mask], mu_1_pred[pred_mask] = self._impute_outcomes(
                        X, None, None, None, pred_mask, idx
                    )

                if self.pseudo_type == "DR" or self.pseudo_type == "PW" or self.pseudo_type == "R":
                    p_pred[pred_mask] = self._impute_propensity(X, None, None, pred_mask, idx)
                else:
                    p_pred[pred_mask] = np.nan

        return mu_0_pred, mu_1_pred, p_pred if (self.fit_propensity_estimator or p is None) else p

    def _fit_and_impute_nuisance_components(self, X, y, w, p=None):
        # fit nuisance components and then save
        X = _get_values_only(X)
        n = len(y)

        # STEP 1: fit plug-in estimators via cross-fitting
        if self.n_folds == 1:
            pred_mask = np.ones(n, dtype=bool)
            self._fold_pred_masks[0] = pred_mask
            # fit plug-in models
            if not self.pseudo_type == "PW":
                mu_0_pred, mu_1_pred = self._impute_outcomes(X, y, w, pred_mask, pred_mask, 0)
            else:
                mu_0_pred, mu_1_pred = None, None

            if self.pseudo_type == "DR" or self.pseudo_type == "PW" or self.pseudo_type == "R":
                p_pred = self._impute_propensity(X, w, pred_mask, pred_mask, 0)
            else:
                p_pred = np.nan

        else:
            if self.pre_cv_po and self.est_params is not None:
                self._do_po_cv(X, y, w)

            mu_0_pred, mu_1_pred, p_pred = np.zeros(n), np.zeros(n), np.zeros(n)

            # create folds stratified by treatment assignment to ensure balance
            splitter = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
            idx = 0
            for train_index, test_index in splitter.split(X, w):
                # create masks
                pred_mask = np.zeros(n, dtype=bool)
                pred_mask[test_index] = 1
                self._fold_pred_masks[idx] = pred_mask

                # fit plug-in te_estimator
                if not self.pseudo_type == "PW":
                    mu_0_pred[pred_mask], mu_1_pred[pred_mask] = self._impute_outcomes(
                        X, y, w, ~pred_mask, pred_mask, idx
                    )

                if self.pseudo_type == "DR" or self.pseudo_type == "PW" or self.pseudo_type == "R":
                    p_pred[pred_mask] = self._impute_propensity(X, w, ~pred_mask, pred_mask, idx)
                else:
                    p_pred[pred_mask] = np.nan

                idx += 1

        return mu_0_pred, mu_1_pred, p_pred if (self.fit_propensity_estimator or p is None) else p

    def _impute_outcomes(self, X, y, w, fit_mask, pred_mask, fold_idx):
        if not self._models_fitted:
            # split sample
            X_fit, Y_fit, W_fit = X[fit_mask, :], y[fit_mask], w[fit_mask]

            if isinstance(self.po_model, BasePluginCATEEstimator):
                # allows first stage model to be e.g. S- or T-learner
                temp_model = clone(self.po_model)
                if self.est_params is not None:
                    temp_model.set_params(est_params=self.est_params)
                temp_model.fit(X_fit, Y_fit, W_fit)
                self._fold_models_po[fold_idx] = temp_model

            else:
                # fit two separate (standard) models
                # untreated model
                temp_model_0 = clone(self.po_model)

                if self.est_params is not None:
                    if isinstance(self.po_model, GridSearchCV):
                        temp_model_0 = clone(self.po_model.estimator)
                    temp_model_0.set_params(**self.est_params[0])
                elif self.pre_cv_po:
                    temp_model_0.set_params(**self.po_params[0])

                temp_model_0.fit(X_fit[W_fit == 0], Y_fit[W_fit == 0])

                # treated model
                temp_model_1 = clone(self.po_model)
                if self.est_params is not None:
                    if isinstance(self.po_model, GridSearchCV):
                        temp_model_1 = clone(self.po_model.estimator)
                    if len(self.est_params) == 1:
                        temp_model_1.set_params(**self.est_params[0])
                    else:
                        temp_model_1.set_params(**self.est_params[1])
                elif self.pre_cv_po:
                    temp_model_1.set_params(**self.po_params[1])

                temp_model_1.fit(X_fit[W_fit == 1], Y_fit[W_fit == 1])
                self._fold_models_po[fold_idx] = [temp_model_0, temp_model_1]

        if isinstance(self.po_model, BasePluginCATEEstimator):
            _, mu_0_pred, mu_1_pred = self._fold_models_po[fold_idx].predict(X[pred_mask, :], return_po=True)
        else:
            if self.binary_y:
                mu_0_pred = self._fold_models_po[fold_idx][0].predict_proba(X[pred_mask, :])
                mu_1_pred = self._fold_models_po[fold_idx][1].predict_proba(X[pred_mask, :])

                if mu_0_pred.ndim > 1:
                    if mu_0_pred.shape[1] == 2:
                        mu_0_pred = mu_0_pred[:, 1]
                        mu_1_pred = mu_1_pred[:, 1]
            else:
                mu_0_pred = self._fold_models_po[fold_idx][0].predict(X[pred_mask, :])
                mu_1_pred = self._fold_models_po[fold_idx][1].predict(X[pred_mask, :])

        return mu_0_pred, mu_1_pred

    def _impute_propensity(self, X, w, fit_mask, pred_mask, fold_idx):
        if self.fit_propensity_estimator:
            if not self._models_fitted:
                # split sample
                X_fit, W_fit = X[fit_mask, :], w[fit_mask]

                # fit propensity estimator
                temp_propensity_estimator = clone(self.prop_model)
                temp_propensity_estimator.fit(X_fit, W_fit)

                # store estimator
                self._fold_models_prop[fold_idx] = temp_propensity_estimator

            # predict propensity on hold out
            p_pred = self._fold_models_prop[fold_idx].predict_proba(X[pred_mask, :])

            if p_pred.ndim > 1:
                if p_pred.shape[1] == 2:
                    p_pred = p_pred[:, 1]

            return p_pred
        else:
            return np.nan


class RTEScorer(PseudoOutcomeTEScorer):
    def _score(self, estimator, X, y_factual, w_factual, p_true=None, t_true=None, sample_weight=None, t_score=None):
        # override parent class: we need different loss function

        t_pred = estimator.predict(X, return_po=False) if t_score is None else t_score

        if not self.plugin_prefit and not self._models_fitted:
            # fit nuisance models
            mu, _, prop = self._fit_and_impute_nuisance_components(X, y_factual, w_factual, p=p_true)
            self._models_fitted = True
        elif not self.plugin_prefit and self._models_fitted:
            # use nuisance models from storage
            mu, _, prop = self._impute_nuisance_components(X, p=p_true)
        else:
            # use prefit model
            mu = self.po_model.predict(X)
            if p_true is not None:
                prop = p_true
            else:
                prop = self.prop_model.predict_proba(X)
                if prop.ndim > 1:
                    if prop.shape[1] == 2:
                        prop = prop[:, 1]

        # construct loss function from mu and prop
        if sample_weight is None:
            return np.mean(((y_factual - mu) - t_pred * (w_factual - prop)) ** 2)
        else:
            return np.mean(sample_weight * ((y_factual - mu) - t_pred * (w_factual - prop)) ** 2) / np.sum(
                sample_weight
            )

    def _impute_outcomes(self, X, y, w, fit_mask, pred_mask, fold_idx):
        # overwrite parent class: for RLearner we need unconditional mean
        if not self._models_fitted:
            X_fit, Y_fit = X[fit_mask, :], y[fit_mask]

            # fit model
            temp_model = clone(self.po_model)
            if self.est_params is not None:
                if isinstance(self.po_model, GridSearchCV):
                    temp_model = clone(self.po_model.estimator)
                temp_model.set_params(**self.est_params[0])
            temp_model.fit(X_fit, Y_fit)
            self._fold_models_po[fold_idx] = temp_model

        if self.binary_y:
            mu_pred = self._fold_models_po[fold_idx].predict_proba(X[pred_mask, :])

            if mu_pred.shape[1] == 2:
                mu_pred = mu_pred[:, 1]
        else:
            mu_pred = self._fold_models_po[fold_idx].predict(X[pred_mask, :])

        return mu_pred, np.ones_like(mu_pred) * np.nan

    def _do_po_cv(self, X, y, w):
        self.po_params = []
        temp_model_0 = GridSearchCV(self.po_model, param_grid=self.grid_po, cv=self.n_folds)
        temp_model_0.fit(X, y)
        self.po_params.append(temp_model_0.best_params_)
        self.po_model.set_params(**self.po_params[0])

    def fit_plugin_model(self, X, y_factual, w_factual):
        # override parent class to allow single outcome model
        if not self.plugin_prefit:
            return
        else:
            if self.est_params is not None:
                if isinstance(self.po_model, GridSearchCV):
                    self.po_model = clone(self.po_model.estimator)
                self.po_model.set_params(**self.est_params[0])
            self.po_model.fit(X, y_factual)
            self.prop_model.fit(X, w_factual)
            self._models_fitted = True
