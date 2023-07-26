"""
Pseudo-Outcome transformations for CATE
"""
# Author: Alicia Curth
import numpy as np


def dr_transformation_cate(y, w, p, mu_0, mu_1):
    """
    Transforms data to efficient influence function/aipw pseudo-outcome for CATE estimation

    Parameters
    ----------
    y : array-like of shape (n_samples,) or (n_samples, )
        The observed outcome variable
    w: array-like of shape (n_samples,)
        The observed treatment indicator
    p: array-like of shape (n_samples,)
        The treatment propensity, estimated or known. Can be None, then p=0.5 is assumed
    mu_0: array-like of shape (n_samples,)
        Estimated or known potential outcome mean of the control group
    mu_1: array-like of shape (n_samples,)
        Estimated or known potential outcome mean of the treatment group
    Returns
    -------
    d_hat:
        EIF/DR transformation for CATE
    """
    if p is None:
        # assume equal
        p = np.full(len(y), 0.5)

    w_1 = w / p
    w_0 = (1 - w) / (1 - p)
    return (w_1 - w_0) * y + ((1 - w_1) * mu_1 - (1 - w_0) * mu_0)


def pw_transformation_cate(y, w, p=None, mu_0=None, mu_1=None):
    """
    Transform data to Horvitz-Thompson transformation for CATE
    Parameters
    ----------
    y : array-like of shape (n_samples,) or (n_samples, )
        The observed outcome variable
    w: array-like of shape (n_samples,)
        The observed treatment indicator
    p: array-like of shape (n_samples,)
        The treatment propensity, estimated or known. Can be None, then p=0.5 is assumed
    mu_0: array-like of shape (n_samples,)
         Estimated or known potential outcome mean of the control group. Placeholder, not used.
    mu_1: array-like of shape (n_samples,)
        Estimated or known potential outcome mean of the treatment group. Placeholder, not used.
    Returns
    -------
    res: array-like of shape (n_samples,)
        Horvitz-Thompson transformed data
    """
    if p is None:
        # assume equal propensities
        p = np.full(len(y), 0.5)
    return (w / p - (1 - w) / (1 - p)) * y


def ra_transformation_cate(y, w, p, mu_0, mu_1):
    """
    Transform data to regression adjustment for CATE

    Parameters
    ----------
    y : array-like of shape (n_samples,) or (n_samples, )
        The observed outcome variable
    w: array-like of shape (n_samples,)
        The observed treatment indicator
    p: array-like of shape (n_samples,)
        Placeholder, not used. The treatment propensity, estimated or known.
    mu_0: array-like of shape (n_samples,)
         Estimated or known potential outcome mean of the control group
    mu_1: array-like of shape (n_samples,)
        Estimated or known potential outcome mean of the treatment group

    Returns
    -------
    res: array-like of shape (n_samples,)
        Regression adjusted transformation
    """
    return w * (y - mu_0) + (1 - w) * (mu_1 - y)


def u_transformation_cate(y, w, p, mu):
    """
    Transform data to U-transformation (described in Kuenzel et al, 2019, Nie & Wager, 2017)
    which underlies both R-learner and U-learner

    Parameters
    ----------
    y : array-like of shape (n_samples,) or (n_samples, )
        The observed outcome variable
    w: array-like of shape (n_samples,)
        The observed treatment indicator
    p: array-like of shape (n_samples,)
        Placeholder, not used. The treatment propensity, estimated or known.
    mu_0: array-like of shape (n_samples,)
         Estimated or known potential outcome mean of the control group
    mu_1: array-like of shape (n_samples,)
        Estimated or known potential outcome mean of the treatment group

    Returns
    -------
    res: array-like of shape (n_samples,)
        Regression adjusted transformation
    """
    return (y - mu) / (w - p)


def pseudo_outcome_transformation(y, w, p, mu_0, mu_1, pseudo_type="DR"):
    if pseudo_type == "DR":
        return dr_transformation_cate(y, w, p, mu_0, mu_1)
    elif pseudo_type == "RA":
        return ra_transformation_cate(y, w, p, mu_0, mu_1)
    elif pseudo_type == "PW":
        return pw_transformation_cate(y, w, p, mu_0, mu_1)
    elif pseudo_type == "U":
        return u_transformation_cate(y, w, p, mu_0, mu_1)
    else:
        raise ValueError("Pseudo outcome type {} was not recognised".format(pseudo_type))
