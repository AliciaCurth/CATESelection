import pandas as pd


def _get_values_only(X):
    # wrapper to return only values of data frame
    if isinstance(X, pd.DataFrame):
        X = X.values
    return X


# some utils
def _check_is_callable(input, name: str = ""):  # pylint: disable=redefined-builtin
    if callable(input):
        pass
    else:
        raise ValueError(
            "Input {} needs to be a callable function so it can " "be used to create simulation.".format(name)
        )
