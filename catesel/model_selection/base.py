""""
Implementations of different scorer classes for CATE estimation. This code mimics
sklearns' scoring modules in sklearn.metrics._scoring
"""
# Author: Alicia Curth
import abc

from sklearn import clone


class _BaseTEScorer:
    def __init__(self, score_func, sign):
        self._score_func = score_func
        self._sign = sign

    def __call__(self, estimator, X, y_factual, w_factual, p_true=None, t_true=None, sample_weight=None, t_score=None):
        """Evaluate predicted target values for X relative to y_factual or t_true

        Parameters
        ----------
        estimator : object
            Trained estimator to use for scoring. Must have a predict_proba
            method; the output of that is used to compute the score.
        X : {array-like, sparse matrix}
            Test data that will be fed to estimator.predict.
        y_factual : array-like
            Factual outcomes observed in sample
        w_factual: array-like
            Factual treatment assignments observed in sample
        p_true: array-like
            Known propensity score
        t_true: array-like
            Known (oracle) treatment effect
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """
        return self._score(
            estimator, X, y_factual, w_factual, p_true, t_true, sample_weight=sample_weight, t_score=t_score
        )

    @abc.abstractmethod
    def _score(self, estimator, X, y_factual, w_factual, p_true=None, t_true=None, sample_weight=None, t_score=None):
        pass

    def reset_models(self):
        if hasattr(self, "po_model"):
            if self.po_model is not None:
                self.po_model = clone(self.po_model)
        if hasattr(self, "prop_model"):
            if self.prop_model is not None:
                self.prop_model = clone(self.prop_model)

        if hasattr(self, "_models_fitted"):
            self._models_fitted = False
