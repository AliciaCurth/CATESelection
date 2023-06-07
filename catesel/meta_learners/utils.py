"""
Utils for meta-learners
"""
# Author: Alicia Curth
from sklearn import clone


def get_name_needed_prediction_method(binary_y: bool):
    """
    Helper function to get the name of 'predict' function in sklearn
    Parameters
    ----------
    binary_y: bool
        Whether the outcome is binary or not

    Returns
    -------
    method name (str)
    """
    if binary_y:
        return 'predict_proba'
    else:
        return 'predict'


def check_estimator_has_method(estimator, needed_method: str, estimator_name: str,
                               return_clone: bool = False):
    """
    Check that an estimator has a specific method.

    Parameters
    ----------
    estimator:
        Estimator to be checked
    needed_method: str
        Name of method
    estimator_name:
        Name of estimator
    return_clone: bool, default False
        Whether to clone the estimator

    Returns
    -------
    clone of estimator if return_clone; else nothing
    """
    if hasattr(estimator, needed_method):
        if return_clone:
            return clone(estimator)
        else:
            pass
    else:
        raise ValueError('{} needs to implement method {}'.format(estimator_name,
                                                                  needed_method))
