"""
Base for CATE meta-learners
"""
# Author: Alicia Curth
import abc
import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import GridSearchCV

from ..meta_learners.utils import get_name_needed_prediction_method,  \
    check_estimator_has_method
from ..utils.base import _get_values_only
from ..utils.weight_utils import compute_importance_weights


class BaseCATEEstimator(BaseEstimator, RegressorMixin, abc.ABC):
    """
    Base class for treatment effect models
    """
    def __init__(self):
        pass

    def score(self, X, y, sample_weight=None):
        pass

    @abc.abstractmethod
    def fit(self, X, y, w, p=None):
        """
        Fit method for a CATEModel

        Parameters
        ----------
        X: pd.DataFrame or np.array
            Covariate matrix
        y: np.array
            Outcome vector
        w: np.array
            Treatment indicator
        p: np.array
            Vector of treatment propensities.
        """
        pass

    @abc.abstractmethod
    def predict(self, X, return_po: bool = False):
        """
        Predict treatment effect estimates using a CATEModel. Depending on method, can also return
        potential outcome estimate.

        Parameters
        ----------
        X: pd.DataFrame or np.array
            Covariate matrix
        return_po: bool, default False
            Whether to return potential outcome estimate

        Returns
        -------
        array of CATE estimates, optionally also potential outcomes and propensity
        """
        pass

    @staticmethod
    def _check_inputs(w, p):
        if p is not None:
            if np.sum(p > 1) > 0 or np.sum(p < 0) > 0:
                raise ValueError('p should be in [0,1]')

        if not ((w == 0) | (w == 1)).all():
            raise ValueError('W should be binary')


class BasePluginCATEEstimator(BaseCATEEstimator):
    """
    Base class for plug-in/ indirect estimators of CATE; such as S- and T-learners

    Parameters
    ----------
    po_estimator: estimator
        Estimator to be used for potential outcome regressions. Should be sklearn-style estimator
        with .fit and .predict/.predict_proba method
    binary_y: bool, default False
        Whether the outcome data is binary
    propensity_estimator: estimator, default None
        Estimator to be used for propensity score estimation (if needed)
    est_params: dict or list of dicts, default None
        Hyperparameters to be passed to po-estimator
    """
    def __init__(self, po_estimator, binary_y: bool = False,
                 propensity_estimator: bool = None,
                 weighting_strategy: str = None,
                 weight_args: dict = None,
                 est_params: list = None):
        self.po_estimator = po_estimator
        self.binary_y = binary_y
        self.propensity_estimator = propensity_estimator
        self.weighting_strategy = weighting_strategy
        self.weight_args = weight_args
        self.est_params = est_params

    def _fit_propensity_estimator(self, X, w):
        if self.propensity_estimator is None:
            raise ValueError("Can only fit propensity estimator if propensity_estimator is not "
                             "None.")
        self.propensity_estimator.fit(X, w)

    def _get_importance_weights(self, X, w):
        p_pred = self.propensity_estimator.predict_proba(X)
        if p_pred.ndim > 1:
            if p_pred.shape[1] == 2:
                p_pred = p_pred[:, 1]
        return compute_importance_weights(p_pred, w, self.weighting_strategy, self.weight_args,
                                          normalize=True)


class TLearner(BasePluginCATEEstimator):
    """
    T-learner for treatment effect estimation (Two learners, fit separately)
    """
    def _prepare_self(self):
        needed_pred_method = get_name_needed_prediction_method(self.binary_y)
        check_estimator_has_method(self.po_estimator, needed_pred_method,
                                   'po_estimator', return_clone=False)

        # to make sure that we are starting with clean objects
        self._plug_in_0 = clone(self.po_estimator)
        self._plug_in_1 = clone(self.po_estimator)

        if self.est_params is not None:
            if isinstance(self.po_estimator, GridSearchCV):
                self._plug_in_1 = clone(self.po_estimator.estimator)
                self._plug_in_0 = clone(self.po_estimator.estimator)

            if len(self.est_params) == 1:
                self._plug_in_1.set_params(**self.est_params[0])
                self._plug_in_0.set_params(**self.est_params[0])
            else:
                self._plug_in_1.set_params(**self.est_params[1])
                self._plug_in_0.set_params(**self.est_params[0])

    def fit(self, X, y, w, p=None):
        """
        Fit plug-in models.

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
        if len(w.shape) > 1:
            w = w.reshape(-1, ).copy()
        if self.weighting_strategy is None:
            # standard T-learner
            self._plug_in_0.fit(X[w == 0], y[w == 0])
            self._plug_in_1.fit(X[w == 1], y[w == 1])

        else:
            # use reweighting within plug-in model
            self._fit_propensity_estimator(X, w)
            weights = self._get_importance_weights(X, w)
            self._plug_in_0.fit(X[w == 0], y[w == 0], sample_weights=weights[w == 0])
            self._plug_in_1.fit(X[w == 1], y[w == 1], sample_weights=weights[w == 1])

    def predict(self, X, return_po: bool = False):
        """
        Predict treatment effects and potential outcomes
        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            Test-sample features
        return_po: bool, default False
            Whether to return potential outcome predictions

        Returns
        -------
        te_est: array-like of shape (n_samples,)
            Predicted treatment effects
        y_0: array-like of shape (n_samples,)
            Predicted Y(0)
        y_1: array-like of shape (n_samples,)
            Predicted Y(1)
        """
        if self.binary_y:
            y_0 = self._plug_in_0.predict_proba(X)
            y_1 = self._plug_in_1.predict_proba(X)

            if y_0.ndim > 1:
                if y_0.shape[1] == 2:
                    y_0 = y_0[:, 1]
                    y_1 = y_1[:, 1]
        else:
            y_0 = self._plug_in_0.predict(X)
            y_1 = self._plug_in_1.predict(X)

        te_est = y_1 - y_0
        if return_po:
            return te_est, y_0, y_1
        else:
            return te_est


class SLearner(BasePluginCATEEstimator):
    """
    S-learner for treatment effect estimation (single learner, treatment indicator just another
    feature).
    """
    def __init__(self, po_estimator, binary_y: bool = False,
                 propensity_estimator: bool = None,
                 weighting_strategy: str = None,
                 weight_args: dict = None,
                 extend_covs: bool = False,
                 est_params: list = None
                 ):
        self.po_estimator = po_estimator
        self.binary_y = binary_y
        self.propensity_estimator = propensity_estimator
        self.weighting_strategy = weighting_strategy
        self.weight_args = weight_args
        self.extend_covs = extend_covs
        self.est_params = est_params

    def _prepare_self(self):
        needed_pred_method = get_name_needed_prediction_method(self.binary_y)
        check_estimator_has_method(self.po_estimator, needed_pred_method,
                                   'po_estimator', return_clone=False)

        # to make sure that we are starting with clean objects
        self._plug_in = clone(self.po_estimator)

        if self.est_params is not None:
            if isinstance(self.po_estimator, GridSearchCV):
                self._plug_in = clone(self.po_estimator.estimator)

            self._plug_in.set_params(**self.est_params[0])

    def fit(self, X, y, w, p=None):
        """
        Fit plug-in models.

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

        # add indicator as additional variable
        X = _get_values_only(X)
        if not self.extend_covs:
            X_ext = np.concatenate((X, w.reshape((-1, 1))), axis=1)
        else:
            # use extended specification
            X_ext = np.concatenate((X, w.reshape((-1, 1)), X * w[:, np.newaxis]), axis=1)

        if self.weighting_strategy is None:
            # fit standard S-learner
            self._plug_in.fit(X_ext, y)
        else:
            # use reweighting within plug-in model
            self._fit_propensity_estimator(X, w)
            weights = self._get_importance_weights(X, w)
            self._plug_in.fit(X_ext, y, sample_weights=weights)

    def predict(self, X, return_po: bool = False):
        """
        Predict treatment effects and potential outcomes

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            Test-sample features
        return_po: bool, default False
            Whether to return potential outcome predictions

        Returns
        -------
        te_est: array-like of shape (n_samples,)
            Predicted treatment effects
        y_0: array-like of shape (n_samples,)
            Predicted Y(0)
        y_1: array-like of shape (n_samples,)
            Predicted Y(1)
        """
        n = X.shape[0]
        X = _get_values_only(X)

        # create extended matrices
        w_1 = np.ones((n, 1))
        w_0 = np.zeros((n, 1))
        if not self.extend_covs:
            X_ext_0 = np.concatenate((X, w_0), axis=1)
            X_ext_1 = np.concatenate((X, w_1), axis=1)
        else:
            X_ext_0 = np.concatenate((X, w_0, X*w_0), axis=1)
            X_ext_1 = np.concatenate((X, w_1, X*w_1), axis=1)

        if self.binary_y:
            y_0 = self._plug_in.predict_proba(X_ext_0)
            y_1 = self._plug_in.predict_proba(X_ext_1)

            if y_0.ndim > 1:
                if y_0.shape[1] == 2:
                    y_0 = y_0[:, 1]
                    y_1 = y_1[:, 1]
        else:
            y_0 = self._plug_in.predict(X_ext_0)
            y_1 = self._plug_in.predict(X_ext_1)

        te_est = y_1 - y_0
        if return_po:
            return te_est, y_0, y_1
        else:
            return te_est