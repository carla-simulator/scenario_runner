#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
The global ParameterDeclarations of an OpenSCENARIO file.

They are parsed once from the .xosc and read back by name wherever a '$name'
reference appears, so they live for as long as the scenario does.
"""

_global_osc_parameters = {}


def update_osc_global_params(parameters):
    """
    Updates/initializes global osc parameters.
    """
    _global_osc_parameters.update(parameters)


def get_osc_global_param_value(ref):
    """
    Returns updated global osc parameter value.
    """
    return _global_osc_parameters.get(ref.replace("$", ""))


def clear_osc_global_params():
    """
    Forgets every parameter, so the next scenario starts from a clean state.
    """
    _global_osc_parameters.clear()
