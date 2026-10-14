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


class VariableSetAction(Action):
    action_type = "VariableSetAction"

    def __init__(self, variable_ref, value, source=None):
        self.variable_ref = variable_ref
        self.value = value
        self.source = source


class VariableModifyAction(Action):
    action_type = "VariableModifyAction"

    def __init__(self, variable_ref, rule, value, source=None):
        self.variable_ref = variable_ref
        self.rule = rule
        self.value = value
        self.source = source


class RouteAction(Action):
    """Base for routing actions retaining OSC semantics."""


class AssignRouteAction(RouteAction):
    action_type = "AssignRouteAction"

    def __init__(self, route=None, catalog_reference=None, source=None):
        self.route = route
        self.catalog_reference = catalog_reference
        self.source = source


class AcquirePositionAction(RouteAction):
    action_type = "AcquirePositionAction"

    def __init__(self, position, source=None):
        self.position = position
        self.source = source


class FollowTrajectoryAction(RouteAction):
    """``trajectory`` or ``catalog_reference`` come from ``TrajectoryRef`` or the deprecated direct children."""

    action_type = "FollowTrajectoryAction"

    def __init__(self, trajectory=None, catalog_reference=None, time_reference=None,
                 following_mode=None, initial_distance_offset=None, source=None):
        self.trajectory = trajectory
        self.catalog_reference = catalog_reference
        self.time_reference = time_reference
        self.following_mode = following_mode
        self.initial_distance_offset = initial_distance_offset
        self.source = source


class SpeedProfileAction(Action):
    """Entries are ``(speed, time)`` lexical pairs; ``time`` is None when omitted."""

    action_type = "SpeedProfileAction"

    def __init__(self, entries, following_mode, entity_ref=None, dynamic_constraints=None, source=None):
        self.entries = list(entries or [])
        self.following_mode = following_mode
        self.entity_ref = entity_ref
        self.dynamic_constraints = dict(dynamic_constraints or {})
        self.source = source
