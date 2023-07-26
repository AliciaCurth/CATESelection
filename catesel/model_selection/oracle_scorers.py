""""
Oracle scorers (require access to usually unknown information; used as baselines)
"""
# Author: Alicia Curth
import numpy as np

from catesel.meta_learners.transformations import pseudo_outcome_transformation
from catesel.model_selection.base import _BaseTEScorer


class OracleTEScorer(_BaseTEScorer):
    # Oracle CATE estimator scorer: scores BaseCATEEstimators on their oracle CATE estimation
    # performance (not available in practice)
    def _score(self, estimator, X, y_factual, w_factual, p_true=None, t_true=None, sample_weight=None, t_score=None):
        if t_true is None:
            raise ValueError("An oracle scorer needs t_true to be known.")

        # get predicted CATE
        t_pred = estimator.predict(X, return_po=False) if t_score is None else t_score

        if sample_weight is not None:
            return self._sign * self._score_func(
                t_true,
                t_pred,
                sample_weight=sample_weight,
            )
        else:
            return self._sign * self._score_func(t_true, t_pred)


class OraclePOScorer(_BaseTEScorer):
    # Oracle CATE estimator scorer: scores BaseCATEEstimators on their oracle PO estimation
    # performance (not available in practice)
    def _score(self, estimator, X, y_factual, w_factual, p_true=None, t_true=None, sample_weight=None, t_score=None):
        # get predicted POs
        if t_score is not None:
            return np.nan

        try:
            _, mu0, mu1 = estimator.predict(X, return_po=True)
        except ValueError:
            return np.nan

        if sample_weight is not None:
            # score by PO performance
            return self._sign * (
                self._score_func(
                    t_true[0],  # pyright: ignore
                    mu0,
                    sample_weight=sample_weight,
                )
                + self._score_func(
                    t_true[1],  # pyright: ignore
                    mu1,
                    sample_weight=sample_weight,
                )
            )
        else:
            return self._sign * (self._score_func(t_true[0], mu0) + self._score_func(t_true[1], mu1))  # pyright: ignore


class OraclePseudoOutcomeScorer(_BaseTEScorer):
    # CATE estimator scorer based on pseudo-outcomes with partial oracle knowledge: compute pseudo
    # outcomes with  oracle knowledge of  nuisance parameters
    def __init__(self, score_func, sign, pseudo_type="DR"):
        super().__init__(score_func, sign)
        self.pseudo_type = pseudo_type

    def _score(
        self,
        estimator,
        X,
        y_factual,
        w_factual,
        p_true=None,
        t_true=None,
        sample_weight=None,
        t_score=None,
        mu0_true=None,
        mu1_true=None,
    ):
        if self.pseudo_type == "DR":
            assert p_true is not None and mu1_true is not None and mu0_true is not None
        elif self.pseudo_type == "RA":
            assert mu1_true is not None and mu0_true is not None
        elif self.pseudo_type == "PW":
            assert p_true is not None

        t_pred = estimator.predict(X, return_po=False) if t_score is None else t_score

        t_pseudo = pseudo_outcome_transformation(
            y_factual, w_factual, p_true, mu0_true, mu1_true, pseudo_type=self.pseudo_type
        )

        if sample_weight is not None:
            return self._sign * self._score_func(
                t_pseudo,
                t_pred,
                sample_weight=sample_weight,
            )
        else:
            return self._sign * self._score_func(t_pseudo, t_pred)
