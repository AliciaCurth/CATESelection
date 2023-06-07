"""
Plug-in Scorers: Use a CATE estimate as surrogate
"""
# Author: Alicia Curth

from catesel.model_selection.base import _BaseTEScorer


class PluginTEScorer(_BaseTEScorer):
    # CATE estimator scorer based on plug-in model for CATE estimation: fits some
    # BASECATEEstimator on validation data and scores against that
    def __init__(self, score_func, sign, po_model=None, plugin_prefit=False):
        if po_model is None:
            raise ValueError('Need to specify po_model')
        self.po_model = po_model
        self.plugin_prefit=plugin_prefit
        super().__init__(score_func, sign)
        self._models_fitted = False
        self.est_params = None

    def set_est_params(self, est_params):
        if est_params is not None:
            if isinstance(est_params, list):
                self.est_params = est_params
            elif isinstance(est_params, dict):
                self.est_params = [est_params]
            else:
                raise ValueError('est_params should be a list of dicts or a dict.')

    def fit_plugin_model(self, X, y_factual, w_factual):
        if not self.plugin_prefit:
            return
        else:
            if self.est_params is not None:
                self.po_model.set_params(est_params=self.est_params)
            self.po_model.fit(X, y_factual, w_factual)
            self._models_fitted = True

    def _score(self, estimator, X, y_factual, w_factual, p_true=None, t_true=None,
               sample_weight=None, t_score=None):

        t_pred = estimator.predict(X, return_po=False) if t_score is None else t_score

        if not self.plugin_prefit and not self._models_fitted:
            if self.est_params is not None:
                self.po_model.set_params(est_params=self.est_params)
            self.po_model.fit(X, y_factual, w_factual)
            self._models_fitted = True # ensure that model does not need to be refitted every time
        t_plugin_pred = self.po_model.predict(X, return_po=False)

        if sample_weight is not None:
            return self._sign * self._score_func(t_plugin_pred, t_pred, sample_weight=sample_weight,
                                               )
        else:
            return self._sign * self._score_func(t_plugin_pred, t_pred)
