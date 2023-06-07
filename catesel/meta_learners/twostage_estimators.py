"""
Multi-stage meta-learners for CATE; those based on pseudo-outcome regression and the R-learner
"""
# Author: Alicia Curth
import abc
import numpy as np

from sklearn import clone
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from catesel.meta_learners.base import BaseCATEEstimator, BasePluginCATEEstimator
from catesel.meta_learners.utils import get_name_needed_prediction_method, \
    check_estimator_has_method
from catesel.utils.base import _get_values_only
from catesel.meta_learners.transformations import u_transformation_cate, \
    dr_transformation_cate, \
    ra_transformation_cate, pw_transformation_cate


class BaseTwoStageEst(BaseCATEEstimator):
    """
    Base class for two-stage estimators.

    Parameters
    ----------
    te_estimator: estimator
        Treatment effect estimator for second stage
    po_estimator: estimator, default None
        estimator for first stage. Can be None, then te_estimator is used
    binary_y: bool, default False
        Whether the outcome data is binary
    fit_propensity_estimator: bool, default False
        Whether to fit a propensity model. Needed for R-learner, DR-learner, PW-learner if
        propensity scores not known
    propensity_estimator: estimator, default None
        estimator for propensity scores. Needed only if fit_propensity_estimator is True.
    n_folds: int, default 1
        Number of cross-fitting folds. If 1, no cross-fitting
    avg_fold_models: bool = False
        if n_folds > 1, should CATE n_fold second stage CATE models be fit and then averaged
        over (if True) or should we fit a single second stage CATE model across all folds (if False)
    est_params: dict or list of dicts, default None
        Hyperparameters to be passed to po-estimator
    random_state: int, default 42
        random state to use for cross-fitting splits
    pre_cv_po: bool, default False
        Do a cv-search of hyperparameters for po-model in grid_po before cross-fitting final models?
    pre_cv_te: bool, default False
        Do a cv-search of hyperparameters for te-model in grid_te before fitting final second
        stage CATE model?
    n_cv_pre: int, default 5
        Number of folds to use in pre_cv_po or pre_cv_te search
    grid_po: dict, default None
        hyperparameter grid for po-model to use in pre_cv_po search
    grid_te: dict, default None
        hyperparameter grid for te-model to use in pre_cv_te search
    """

    def __init__(self, te_estimator,
                 po_estimator=None,
                 propensity_estimator=None,
                 binary_y: bool = False,
                 fit_propensity_estimator: bool = True,
                 n_folds: int = 1,
                 avg_fold_models=False,
                 est_params=None,
                 random_state: int = 42,
                 pre_cv_po=False,
                 pre_cv_te=False,
                 n_cv_pre=5,
                 grid_po=None,
                 grid_te=None):
        # set estimators
        self.te_estimator = te_estimator
        self.po_estimator = po_estimator

        self.fit_propensity_estimator = fit_propensity_estimator
        self.propensity_estimator = propensity_estimator

        # set other arguments
        self.n_folds = n_folds
        self.random_state = random_state
        self.binary_y = binary_y
        self.est_params = est_params

        # set arguments about internal cv for internal models
        self.pre_cv_po = pre_cv_po
        self.pre_cv_te = pre_cv_te
        self.n_cv_pre = n_cv_pre
        self.grid_po = grid_po
        self.grid_te = grid_te
        self.po_params = None
        self.avg_fold_models = avg_fold_models
        self.splitter = None
        self.te_params = None

    def fit(self, X, y, w, p=None):
        """
        Fit two stages of pseudo-outcome regression to get treatment effect estimators
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The features to fit to
        y : array-like of shape (n_samples,) or (n_samples, )
            The outcome variable
        w: array-like of shape (n_samples,)
            The treatment indicator
        p: array-like of shape (n_samples,)
            The treatment propensity
        """
        self._prepare_self()
        self._check_inputs(w, p)
        self._fit(X, y, w, p)

    def fit_and_impute_nuisance_components_only(self, X, y, w, p=None):
        X = _get_values_only(X)
        n = len(y)

        # STEP 1: fit plug-in estimators via cross-fitting
        if self.n_folds == 1:
            pred_mask = np.ones(n, dtype=bool)
            # fit plug-in models
            mu_0_pred, mu_1_pred, p_pred = self._first_step(X, y, w, pred_mask, pred_mask)

        else:
            mu_0_pred, mu_1_pred, p_pred = np.zeros(n), np.zeros(n), np.zeros(n)

            # create folds stratified by treatment assignment to ensure balance
            splitter = StratifiedKFold(n_splits=self.n_folds, shuffle=True,
                                       random_state=self.random_state)
            self.splitter = splitter

            for train_index, test_index in splitter.split(X, w):
                # create masks
                pred_mask = np.zeros(n, dtype=bool)
                pred_mask[test_index] = 1

                # fit plug-in te_estimator
                mu_0_pred[pred_mask], mu_1_pred[pred_mask], p_pred[pred_mask] = \
                    self._first_step(X, y, w, ~pred_mask, pred_mask)

        if p is None or self.fit_propensity_estimator is True:
            # use estimated propensity scores
            p = p_pred

        return mu_0_pred, mu_1_pred, p

    def _do_po_cv(self, X, y, w):
        self.po_params = []
        temp_model_0 = GridSearchCV(self.po_estimator, param_grid=self.grid_po, cv=self.n_cv_pre)
        temp_model_0.fit(X[w == 0], y[w == 0])
        self.po_params.append(temp_model_0.best_params_)

        # treated model
        temp_model_1 = GridSearchCV(self.po_estimator, param_grid=self.grid_po, cv=self.n_cv_pre)
        temp_model_1.fit(X[w == 1], y[w == 1])
        self.po_params.append(temp_model_1.best_params_)

    def _fit(self, X, y, w, p=None):
        X = _get_values_only(X)
        n = len(y)

        # do find hyperparameter settings before doing final cross-fitting step
        if self.pre_cv_po and self.est_params is None:
            self._do_po_cv(X, y, w)

        # STEP 1: fit plug-in estimators via cross-fitting
        if self.n_folds == 1:
            pred_mask = np.ones(n, dtype=bool)
            # fit plug-in models
            mu_0_pred, mu_1_pred, p_pred = self._first_step(X, y, w, pred_mask, pred_mask)

        else:
            mu_0_pred, mu_1_pred, p_pred = np.zeros(n), np.zeros(n), np.zeros(n)

            # create folds stratified by treatment assignment to ensure balance
            splitter = StratifiedKFold(n_splits=self.n_folds, shuffle=True,
                                       random_state=self.random_state)

            self._test_idx_list = []

            for train_index, test_index in splitter.split(X, w):
                # create masks
                pred_mask = np.zeros(n, dtype=bool)
                pred_mask[test_index] = 1

                # fit plug-in te_estimator
                mu_0_pred[pred_mask], mu_1_pred[pred_mask], p_pred[pred_mask] = \
                    self._first_step(X, y, w, ~pred_mask, pred_mask)

                self._test_idx_list.append(test_index)

        if p is None or self.fit_propensity_estimator is True:
            # use estimated propensity scores
            p = p_pred

        # STEP 2: direct TE estimation
        self._second_step(X, y, w, p, mu_0_pred, mu_1_pred)

    def predict(self, X, return_po=False):
        """
        Predict treatment effects

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            Test-sample features
        return_po: bool, default False
            Whether to return potential outcome predictions. Placeholder, can only accept False.
        Returns
        -------
        te_est: array-like of shape (n_samples,)
            Predicted treatment effects
        """
        if return_po:
            raise ValueError('Multi-stage estimators return only an estimate of CATE.')

        if not self.avg_fold_models or self.n_folds == 1:
            # there is only a single treatment effect estimator
            return self.te_estimator.predict(X)
        else:
            # there are multiple treatment effect estimators, predict their average
            preds = np.zeros((X.shape[0], self.n_folds))
            for i in range(self.n_folds):
                preds[:, i] = self._te_model_list[i].predict(X)
            return np.average(preds, axis=1)

    @abc.abstractmethod
    def _first_step(self, X, y, w, fit_mask, pred_mask):
        pass

    @abc.abstractmethod
    def _second_step(self, X, y, w, p, mu_0, mu_1):
        pass

    def _impute_pos(self, X, y, w, fit_mask, pred_mask):
        # split sample
        X_fit, Y_fit, W_fit = X[fit_mask, :], y[fit_mask], w[fit_mask]

        if isinstance(self.po_estimator, BasePluginCATEEstimator):
            # allows first stage model to be e.g. S- or T-learner
            temp_model = clone(self.po_estimator)

            if self.est_params is not None:
                temp_model.set_params(est_params=self.est_params)

            temp_model.fit(X_fit, Y_fit, W_fit)
            _, mu_0_pred, mu_1_pred = temp_model.predict(X[pred_mask, :], return_po=True)
        else:
            # fit two separate (standard) models
            # untreated model
            temp_model_0 = clone(self.po_estimator)

            # if we did some hyperparameter tuning
            if self.est_params is not None:
                if isinstance(self.po_estimator, GridSearchCV):
                    temp_model_0 = clone(self.po_estimator.estimator)
                temp_model_0.set_params(**self.est_params[0])
            elif self.pre_cv_po:
                temp_model_0.set_params(**self.po_params[0])

            temp_model_0.fit(X_fit[W_fit == 0], Y_fit[W_fit == 0])

            # treated model
            temp_model_1 = clone(self.po_estimator)

            if self.est_params is not None:
                if isinstance(self.po_estimator, GridSearchCV):
                    temp_model_1 = clone(self.po_estimator.estimator)
                if len(self.est_params) == 1:
                    temp_model_1.set_params(**self.est_params[0])
                else:
                    temp_model_1.set_params(**self.est_params[1])

            elif self.pre_cv_po:
                temp_model_1.set_params(**self.po_params[1])

            temp_model_1.fit(X_fit[W_fit == 1], Y_fit[W_fit == 1])

            if self.binary_y:
                mu_0_pred = temp_model_0.predict_proba(X[pred_mask, :])
                mu_1_pred = temp_model_1.predict_proba(X[pred_mask, :])

                if mu_0_pred.ndim > 1:
                    if mu_0_pred.shape[1] == 2:
                        mu_0_pred = mu_0_pred[:, 1]
                        mu_1_pred = mu_1_pred[:, 1]
            else:
                mu_0_pred = temp_model_0.predict(X[pred_mask, :])
                mu_1_pred = temp_model_1.predict(X[pred_mask, :])

        return mu_0_pred, mu_1_pred

    def _impute_propensity(self, X, w, fit_mask, pred_mask):
        if self.fit_propensity_estimator:
            # split sample
            X_fit, W_fit = X[fit_mask, :], w[fit_mask]

            # fit propensity estimator
            temp_propensity_estimator = clone(self.propensity_estimator)
            temp_propensity_estimator.fit(X_fit, W_fit)

            # predict propensity on hold out
            p_pred = temp_propensity_estimator.predict_proba(X[pred_mask, :])

            if p_pred.ndim > 1:
                if p_pred.shape[1] == 2:
                    p_pred = p_pred[:, 1]

            return p_pred
        else:
            return None

    def _impute_unconditional_mean(self, X, y, fit_mask, pred_mask):
        # R-learner and U-learner need to impute unconditional mean
        X_fit, Y_fit = X[fit_mask, :], y[fit_mask]

        # fit model
        temp_model = clone(self.po_estimator)

        if self.est_params is not None:
            if isinstance(self.po_estimator, GridSearchCV):
                temp_model = clone(self.po_estimator.estimator)
            temp_model.set_params(**self.est_params[0])
        elif self.pre_cv_po:
            temp_model.set_params(**self.po_params)

        temp_model.fit(X_fit, Y_fit)

        if self.binary_y:
            mu_pred = temp_model.predict_proba(X[pred_mask, :])

            if mu_pred.shape[1] == 2:
                mu_pred = mu_pred[:, 1]
        else:
            mu_pred = temp_model.predict(X[pred_mask, :])

        return mu_pred

    def _prepare_self(self):
        if self.po_estimator is None:
            self.po_estimator = clone(self.te_estimator)

        if self.fit_propensity_estimator and self.propensity_estimator is None:
            raise ValueError("Need to pass propensity_estimator if you wish to fit a propensity "
                             "estimator")

        # check that all estimators have the attributes they should have and clone them to be safe
        if self.propensity_estimator is not None:
            self.propensity_estimator = check_estimator_has_method(self.propensity_estimator,
                                                                   'predict_proba',
                                                                   'propensity_estimator',
                                                                   return_clone=True)

        if not isinstance(self.po_estimator, BasePluginCATEEstimator):
            needed_pred_method = get_name_needed_prediction_method(self.binary_y)
            self.po_estimator = check_estimator_has_method(self.po_estimator,
                                                           needed_pred_method,
                                                           'po_estimator', return_clone=True)

        self.te_estimator = check_estimator_has_method(self.te_estimator,
                                                       'predict',
                                                       'te_estimator', return_clone=True)
        if self.pre_cv_po and self.grid_po is None:
            raise ValueError('Can only do pre-cv-po if parameter grid grid_po is specified.')

        if self.pre_cv_te and self.grid_te is None:
            raise ValueError('Can only do pre-cv-te if parameter grid grid_te is specified.')

        if self.est_params is not None:
            if not isinstance(self.est_params, list):
                if isinstance(self.est_params, dict):
                    self.est_params = [self.est_params]
                else:
                    raise ValueError('est_params should be a list of dicts or a dict.')

        self._individual_checks()

    def _individual_checks(self):
        pass


class DRLearner(BaseTwoStageEst):
    """
    DR-learner for CATE estimation, based on doubly robust AIPW pseudo-outcome
    """

    def _first_step(self, X, y, w, fit_mask, pred_mask):
        mu0_pred, mu1_pred = self._impute_pos(X, y, w, fit_mask, pred_mask)
        p_pred = self._impute_propensity(X, w, fit_mask, pred_mask)
        return mu0_pred, mu1_pred, p_pred

    def _second_step(self, X, y, w, p, mu_0, mu_1):
        pseudo_outcome = dr_transformation_cate(y, w, p, mu_0, mu_1)

        if self.n_cv_pre > 1 and self.pre_cv_te:
            te_est_temp = GridSearchCV(self.te_estimator, self.grid_te, cv=self.n_cv_pre)
            te_est_temp.fit(X, pseudo_outcome)
            self.te_params = te_est_temp.best_params_

            self.te_estimator = clone(self.te_estimator)
            self.te_estimator.set_params(**self.te_params)

        if self.avg_fold_models and self.n_folds > 1:
            self._te_model_list = []
            for i in range(self.n_folds):
                temp_est_i = clone(self.te_estimator)
                temp_est_i.fit(X[self._test_idx_list[i], :], pseudo_outcome[self._test_idx_list[i]])
                self._te_model_list.append(temp_est_i)
        else:
            self.te_estimator.fit(X, pseudo_outcome)


class PWLearner(BaseTwoStageEst):
    """
    PW-learner for CATE estimation, based on singly robust Horvitz Thompson pseudo-outcome
    """

    def _first_step(self, X, y, w, fit_mask, pred_mask):
        mu0_pred, mu1_pred = np.nan, np.nan  # not needed
        p_pred = self._impute_propensity(X, w, fit_mask, pred_mask)
        return mu0_pred, mu1_pred, p_pred

    def _second_step(self, X, y, w, p, mu_0, mu_1):
        pseudo_outcome = pw_transformation_cate(y, w, p)
        self.te_estimator.fit(X, pseudo_outcome)


class RALearner(BaseTwoStageEst):
    """
    RA-learner for CATE estimation, based on singly robust regression-adjusted pseudo-outcome
    """
    def _first_step(self, X, y, w, fit_mask, pred_mask):
        mu0_pred, mu1_pred = self._impute_pos(X, y, w, fit_mask, pred_mask)
        p_pred = np.nan  # not needed
        return mu0_pred, mu1_pred, p_pred

    def _second_step(self, X, y, w, p, mu_0, mu_1):
        pseudo_outcome = ra_transformation_cate(y, w, p, mu_0, mu_1)
        self.te_estimator.fit(X, pseudo_outcome)


class VirtualTwin(BaseTwoStageEst):
    """
    Virtual Twin for CATE estimation. Very similar to T-/S-learner, but instead of outputting
    mu1(x)-mu0(x), regresses mu1(x)-mu0(x) on X with a second stage model.
    """
    def _first_step(self, X, y, w, fit_mask, pred_mask):
        mu0_pred, mu1_pred = self._impute_pos(X, y, w, fit_mask, pred_mask)
        p_pred = np.nan  # not needed
        return mu0_pred, mu1_pred, p_pred

    def _second_step(self, X, y, w, p, mu_0, mu_1):
        pseudo_outcome = mu_1 - mu_0
        self.te_estimator.fit(X, pseudo_outcome)


class ULearner(BaseTwoStageEst):
    """
    U-learner for CATE estimation. Based on pseudo-outcome (Y-mu(x))/(w-pi(x))
    """

    def _first_step(self, X, y, w, fit_mask, pred_mask):
        mu_pred = self._impute_unconditional_mean(X, y, fit_mask, pred_mask)
        mu1_pred = np.nan  # only have one thing to impute here
        p_pred = self._impute_propensity(X, w, fit_mask, pred_mask)
        return mu_pred, mu1_pred, p_pred

    def _second_step(self, X, y, w, p, mu_0, mu_1):
        pseudo_outcome = u_transformation_cate(y, w, p, mu_0)
        self.te_estimator.fit(X, pseudo_outcome)


class RLearner(BaseTwoStageEst):
    """
    R-learner for CATE estimation. Based on pseudo-outcome (Y-mu(x))/(w-pi(x)) and sample weight
    (w-pi(x))^2 -- can only be implemented if .fit of te_estimator takes argument 'sample_weight'.
    """

    def _first_step(self, X, y, w, fit_mask, pred_mask):
        mu_pred = self._impute_unconditional_mean(X, y, fit_mask, pred_mask)
        mu1_pred = np.nan  # only have one thing to impute here
        p_pred = self._impute_propensity(X, w, fit_mask, pred_mask)
        return mu_pred, mu1_pred, p_pred

    def _second_step(self, X, y, w, p, mu_0, mu_1):
        pseudo_outcome = u_transformation_cate(y, w, p, mu_0)

        if self.n_cv_pre > 1 and self.pre_cv_te:
            te_est_temp = GridSearchCV(self.te_estimator, self.grid_te, cv=self.n_cv_pre)
            te_est_temp.fit(X, pseudo_outcome, sample_weight=(w - p) ** 2)
            self.te_params = te_est_temp.best_params_

            self.te_estimator = clone(self.te_estimator)
            self.te_estimator.set_params(**self.te_params)

        if self.avg_fold_models and self.n_folds > 1:
            self._te_model_list = []
            for i in range(self.n_folds):
                temp_est_i = clone(self.te_estimator)
                temp_est_i.fit(X[self._test_idx_list[i], :], pseudo_outcome[self._test_idx_list[i]],
                               sample_weight=((w - p) ** 2)[self._test_idx_list[i]])
                self._te_model_list.append(temp_est_i)
        else:
            self.te_estimator.fit(X, pseudo_outcome, sample_weight=(w - p) ** 2)

    def _do_po_cv(self, X, y, w):
        temp_model = GridSearchCV(self.po_estimator, param_grid=self.grid_po, cv=self.n_cv_pre)
        temp_model.fit(X, y)
        self.po_params = temp_model.best_params_


class XLearner(BaseTwoStageEst):
    """
    X-learner for CATE estimation. Combines two CATE estimates via a weighting function g(x):
    tau(x) = g(x) tau_0(x) + (1-g(x)) tau_1(x)
    """

    def __init__(self, te_estimator, po_estimator=None, propensity_estimator=None,
                 binary_y: bool = False, fit_propensity_estimator: bool = True,
                 weighting_strategy='prop'):
        super().__init__(te_estimator=te_estimator, po_estimator=po_estimator,
                         propensity_estimator=propensity_estimator, binary_y=binary_y,
                         fit_propensity_estimator=fit_propensity_estimator, n_folds=1)
        self.weighting_strategy = weighting_strategy

    def _individual_checks(self):
        # check if weighting functions
        if self.weighting_strategy.isnumeric():
            if self.weighting_strategy > 1 or self.weighting_strategy < 0:
                raise ValueError('Numeric weighting_strategy should be in [0, 1]')
            pass
        elif type(self.weighting_strategy) is str:
            if self.weighting_strategy == 'prop':
                pass
            elif self.weighting_strategy == '1-prop':
                pass
            else:
                raise ValueError('Weighting strategy should be numeric, "prop", "1-prop" or a'
                                 'callable.')
        elif callable(self.weighting_strategy):
            pass
        else:
            raise ValueError('Weighting strategy should be numeric, "prop", "1-prop" or a'
                             'callable.')

    def _first_step(self, X, y, w, fit_mask, pred_mask):
        mu0_pred, mu1_pred = self._impute_pos(X, y, w, fit_mask, pred_mask)
        p_pred = np.nan
        return mu0_pred, mu1_pred, p_pred

    def _second_step(self, X, y, w, p, mu_0, mu_1):
        # split by treatment status, fit one model per group
        pseudo_0 = mu_1[w == 0] - y[w == 0]
        self._te_estimator_0 = clone(self.te_estimator)
        self._te_estimator_0.fit(X[w == 0], pseudo_0)

        pseudo_1 = y[w == 1] - mu_0[w == 1]
        self._te_estimator_1 = clone(self.te_estimator)
        self._te_estimator_1.fit(X[w == 1], pseudo_1)

        if self.weighting_strategy == 'prop' or self.weighting_strategy == '1-prop':
            self.propensity_estimator.fit(X, w)

    def predict(self, X, return_po=False):
        """
        Predict treatment effects

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            Test-sample features
        return_po: bool, default False
            Whether to return potential outcome predictions. Placeholder, can only accept False.
        Returns
        -------
        te_est: array-like of shape (n_samples,)
            Predicted treatment effects
        """
        if return_po:
            raise ValueError('Multi-stage estimators return only an estimate of CATE.')

        tau0_pred = self._te_estimator_0.predict(X)
        tau1_pred = self._te_estimator_1.predict(X)

        if self.weighting_strategy == 'prop' or self.weighting_strategy == '1-prop':
            prop_pred = self.propensity_estimator.predict(X)

        if self.weighting_strategy == 'prop':
            weight = prop_pred
        elif self.weighting_strategy == '1-prop':
            weight = 1 - prop_pred
        elif self.weighting_strategy.isnumeric():
            weight = self.weighting_strategy
        else:
            # weighting strategy must be callable
            weight = self.weighting_strategy(X)
        return weight * tau0_pred + (1 - weight) * tau1_pred
