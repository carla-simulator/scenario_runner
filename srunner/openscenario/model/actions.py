"""Typed action objects used by the incremental execution boundary."""

from __future__ import absolute_import


class Action(object):
    """Base class for semantic actions."""

    action_type = "Action"


class ParameterAction(Action):
    """Semantic representation of OSC ``ParameterAction``.

    Values remain lexical at this boundary.  Parameter expression evaluation is
    deliberately left to the parameter subsystem and is not coupled to CARLA.
    """

    action_type = "ParameterAction"

    def __init__(self, parameter_ref, set_value=None, modify_rule=None, modify_value=None, source=None):
        self.parameter_ref = parameter_ref
        self.set_value = set_value
        self.modify_rule = modify_rule
        self.modify_value = modify_value
        self.source = source

    @property
    def is_modify(self):
        return self.modify_rule is not None
