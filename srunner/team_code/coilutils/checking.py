import torch
import h5py
import sys
import numpy as np
import numbers


def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3

def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

def do_assert(condition, message="Assertion failed."):
    """
    Function that behaves equally to an `assert` statement, but raises an
    Exception.

    This is added because `assert` statements are removed in optimized code.
    It replaces `assert` statements throughout the library that should be
    kept even in optimized code.

    Parameters
    ----------
    condition : bool
        If False, an exception is raised.

    message : string, optional(default="Assertion failed.")
        Error message.

    """
    if not condition:
        raise AssertionError(str(message))

def is_single_number(val):
    """
    Checks whether a variable is a number, i.e. an integer or float.

    Parameters
    ----------
    val : anything
        The variable to
        check.

    Returns
    -------
    out : bool
        True if the variable is a number. Otherwise False.

    """
    return isinstance(val, numbers.Integral) or isinstance(val, numbers.Real)


def is_callable(val):
    """
    Checks whether a variable is a callable, e.g. a function.

    Parameters
    ----------
    val : anything
        The variable to
        check.

    Returns
    -------
    out : bool
        True if the variable is a callable. Otherwise False.

    """
    # python 3.x with x <= 2 does not support callable(), apparently
    if sys.version_info[0] == 3 and sys.version_info[1] <= 2:
        return hasattr(val, '__call__')
    else:
        return callable(val)

# TODO Resource temporarily unavailable.

def is_hdf5_prepared(filename):
    """
        We add this checking to verify if the hdf5 file has all the necessary metadata needed for performing,
        our trainings.
        # TODO: I dont know the scope but maybe this can change depending on the system. BUt i want to keep this for
        CARLA

    """

    data = h5py.File(filename, "r+")

    # Check if the number of metadata is correct, the current number is 28


    if len(data['metadata_targets']) < 28:
        return False
    if len(data['targets'][0]) < 28:
        return False


    # Check if the steering is fine
    if sum(data['targets'][0, :]) == 0.0:
        return False


    data.close()
    return True


