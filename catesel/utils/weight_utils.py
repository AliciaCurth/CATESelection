"""
Implement different reweighting/balancing strategies, as in Li et al (2018)
"""
# Author: Alicia Curth
import numpy as np

IPW_NAME = 'ipw'
TRUNC_IPW_NAME = 'truncipw'
OVERLAP_NAME = 'overlap'
MATCHING_NAME = 'match'

ALL_WEIGHTING_STRATEGIES = [IPW_NAME, TRUNC_IPW_NAME, OVERLAP_NAME, MATCHING_NAME]


def compute_importance_weights(propensity, w, weighting_strategy, weight_args: dict = None,
                               normalize: bool = True):
    if weighting_strategy not in ALL_WEIGHTING_STRATEGIES:
        raise ValueError("weighting_strategy should be in "
                         "catesel.utils.weight_utils.ALL_WEIGHTING_STRATEGIES. "
                         "You passed {}".format(weighting_strategy))
    if weight_args is None:
        weight_args = {}

    if weighting_strategy == IPW_NAME:
        return compute_ipw(propensity, w, normalize=normalize)
    elif weighting_strategy == TRUNC_IPW_NAME:
        return compute_trunc_ipw(propensity, w, normalize=normalize, **weight_args)
    elif weighting_strategy == OVERLAP_NAME:
        return compute_overlap_weights(propensity, w, normalize=normalize)
    elif weighting_strategy == MATCHING_NAME:
        return compute_matching_weights(propensity, w, normalize=normalize)


def compute_ipw(propensity, w, normalize):
    if normalize:
        p_hat = np.average(w)
        return w * p_hat / propensity + (1 - w) * (1 - p_hat) / (1 - propensity)
    else:
        return w / propensity + (1 - w)/ (1 - propensity)


def compute_trunc_ipw(propensity, w, cutoff: float = 0, normalize=False):
    if cutoff < 0 or cutoff > 1:
        raise ValueError("cutoff needs to be between 0 and 1.")
    ipw = compute_ipw(propensity, w, normalize=normalize)
    return np.where((propensity > cutoff) & (propensity < 1 - cutoff),
                    ipw, np.where(propensity < cutoff,
                    cutoff, 1 - cutoff))


def compute_matching_weights(propensity, w, normalize=False):
    ipw = compute_ipw(propensity, w, normalize=normalize)
    return np.minimum(ipw, 1-ipw) * ipw


def compute_overlap_weights(propensity, w, normalize=False):
    ipw = compute_ipw(propensity, w, normalize=normalize)
    return propensity * (1 - propensity) * ipw
