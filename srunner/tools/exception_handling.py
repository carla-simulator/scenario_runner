#!/usr/bin/env python

# Copyright (c) 2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides a refined exception handling
"""


class ExceptionHandler(object):

    """
    """

    raise_mode = True

    @staticmethod
    def raise_exception(type, string):

        if ExceptionHandler.raise_mode:
            if type == "RuntimeError":
                raise RuntimeError(string)
            elif type == "ValueError":
                raise ValueError(string)
            elif type == "KeyError":
                raise KeyError(string)
            else:
                raise Exception(string)
