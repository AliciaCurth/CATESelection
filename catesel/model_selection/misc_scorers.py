"""
Other scorers: Matching and Influence functions
"""
# pylint: disable=attribute-defined-outside-init

# Author: Alicia Curth

import numpy as np

from catesel.model_selection.base import _BaseTEScorer
from catesel.model_selection.pseudooutcome_scorers import PseudoOutcomeTEScorer


class MatchScorer(_BaseTEScorer):
    # CATE estimator scorer based on matching: scores BaseCATEEstimators by nearest neighbor
    # matching of two factuals in euclidean distance (based on Rolling & Yang, 2014)
    def __init__(self, score_func, sign):
        super().__init__(score_func, sign)
        self._matches_made = False

    def _score(self, estimator, X, y_factual, w_factual, p_true=None, t_true=None, sample_weight=None, t_score=None):
        t_pred = estimator.predict(X, return_po=False) if t_score is None else t_score

        if not self._matches_made or (t_score is not None):
            self._get_matched_ite(X, y_factual, w_factual)
            if t_score is not None:
                self._matches_made = False

        if sample_weight is not None:
            return self._sign * self._score_func(
                self._matched_effects,
                t_pred,
                sample_weight=sample_weight,
            )
        else:
            return self._sign * self._score_func(self._matched_effects, t_pred)

    def _get_matched_ite(self, X, y_factual, w_factual):
        _matched_effects = np.ones_like(y_factual)
        X_treat = X[w_factual == 1, :]
        y_treat = y_factual[w_factual == 1]
        X_control = X[w_factual == 0, :]
        y_control = y_factual[w_factual == 0]
        for i in range(X.shape[0]):
            # find match
            if w_factual[i] == 1:
                dists = np.sum((X_control - X[i, :]) ** 2, axis=1)
                match_i = np.argmin(dists)
                _matched_effects[i] = y_factual[i] - y_control[match_i]
            else:
                dists = np.sum((X_treat - X[i, :]) ** 2, axis=1)
                match_i = np.argmin(dists)
                _matched_effects[i] = y_treat[match_i] - y_factual[i]

        self._matched_effects = _matched_effects
        self._matches_made = True  # save so don't have to compute every time we score a new
        # estimator

    def reset_models(self):
        self._matched_effects = None
        self._matches_made = False


class IFTEScorer(PseudoOutcomeTEScorer):
    # Influence function scoring, as in Alaa & van der Schaar 2019
    def _score(self, estimator, X, y_factual, w_factual, p_true=None, t_true=None, sample_weight=None, t_score=None):
        t_pred = estimator.predict(X, return_po=False) if t_score is None else t_score

        # fit nuisance component models (as in normal pseudooutcome scorer
        if not self.plugin_prefit and not self._models_fitted:
            # fit nuisance models
            mu0, mu1, prop = self._fit_and_impute_nuisance_components(X, y_factual, w_factual, p=p_true)
            self._models_fitted = True
        elif not self.plugin_prefit and self._models_fitted:
            # use nuisance models from storage
            mu0, mu1, prop = self._impute_nuisance_components(X, p=p_true)
        else:
            # use prefit model
            _, mu0, mu1 = self.po_model.predict(X, return_po=True)  # pyright: ignore
            if p_true is not None:
                prop = p_true
            else:
                if self.prop_model is not None:
                    prop = self.prop_model.predict_proba(X)[:, 1]
                else:
                    raise ValueError("need propensities")

        # from here on do IF correction; note: plug-in estimate cancels out
        A = w_factual - prop
        C = prop * (1 - prop)
        B = 2 * w_factual * (w_factual - prop) / C

        mean_IF = np.mean(
            (1 - B) * (mu1 - mu0) ** 2  # pyright: ignore
            + B * y_factual * (mu1 - mu0 - t_pred)  # pyright: ignore
            - A * (mu1 - mu0 - t_pred) ** 2  # pyright: ignore
            + t_pred**2
        )

        return mean_IF
