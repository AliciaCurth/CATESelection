"""
Factual Scorers for CATE Estimators
"""
# Author: Alicia Curth

import numpy as np

from catesel.model_selection.base import _BaseTEScorer
from catesel.utils.weight_utils import compute_trunc_ipw


class FactualTEScorer(_BaseTEScorer):
    # Factual CATE estimator scorer: scores BaseCATEEstimators on their factual prediction
    # performance
    def _score(self, estimator, X, y_factual, w_factual, p_true=None, t_true=None, sample_weight=None, t_score=None):
        if t_score is not None:
            # cannot score CATE prediction, needs po-predictions
            return np.nan

        # get predicted POs
        try:
            _, mu0, mu1 = estimator.predict(X, return_po=True)
        except ValueError:
            # needs po-predictions
            return np.nan

        y_pred = (1 - w_factual) * mu0 + w_factual * mu1

        if sample_weight is not None:
            return self._sign * self._score_func(
                y_factual,
                y_pred,
                sample_weight=sample_weight,
            )
        else:
            return self._sign * self._score_func(y_factual, y_pred)


class wFactualTEScorer(_BaseTEScorer):
    def __init__(self, score_func, sign, prop_model=None, plugin_prefit=False, cutoff=0):
        self.prop_model = prop_model
        self.plugin_prefit = plugin_prefit
        self.cutoff = cutoff
        self._models_fitted = False
        super().__init__(score_func, sign)

    def fit_plugin_model(self, X, y_factual, w_factual):
        if not self.plugin_prefit:
            return
        else:
            self.prop_model.fit(X, w_factual)  # pyright: ignore
            self._models_fitted = True

    def _score(self, estimator, X, y_factual, w_factual, p_true=None, t_true=None, sample_weight=None, t_score=None):
        # get predicted POs
        if t_score is not None:
            return np.nan

        try:
            _, mu0, mu1 = estimator.predict(X, return_po=True)
        except ValueError:
            return np.nan

        y_pred = (1 - w_factual) * mu0 + w_factual * mu1

        if p_true is not None:
            prop = p_true
        else:
            if not self.plugin_prefit and not self._models_fitted:
                self.prop_model.fit(X, w_factual)  # pyright: ignore
                self._models_fitted = True
            prop = self.prop_model.predict_proba(X)[:, 1]  # pyright: ignore

        sample_weight = compute_trunc_ipw(prop, w_factual, cutoff=self.cutoff, normalize=True)

        return self._sign * self._score_func(
            y_factual,
            y_pred,
            sample_weight=sample_weight,
        )
