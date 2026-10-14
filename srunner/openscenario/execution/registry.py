"""Registry-based dispatch for semantic actions."""

from __future__ import absolute_import

from srunner.openscenario.model.expressions import EvaluationError
from srunner.openscenario.validation.diagnostics import Diagnostic


class UnsupportedFeatureError(RuntimeError):
    """A feature the semantic runtime does not execute.

    The default diagnostic (code ``Diagnostic.NOT_MIGRATED``) means the feature has no semantic
    executor yet and the legacy runtime may handle it.  Any other diagnostic is final.
    """

    def __init__(self, action, diagnostic=None):
        self.action = action
        self.diagnostic = diagnostic or Diagnostic(
            "unsupported_feature", "No executor registered for {}".format(
                getattr(action, "tag", None) or type(action).__name__),
            source=getattr(action, "source", None), code=Diagnostic.NOT_MIGRATED)
        super(UnsupportedFeatureError, self).__init__(str(self.diagnostic))

    @property
    def not_migrated(self):
        return self.diagnostic.code == Diagnostic.NOT_MIGRATED


def backend_capability_error(subject, message):
    """UnsupportedFeatureError for a migrated feature the selected backend cannot execute."""
    return UnsupportedFeatureError(subject, Diagnostic(
        "backend_capability_error", message, source=getattr(subject, "source", None)))


class ActionExecutorRegistry(object):
    def __init__(self):
        self._executors = {}

    def register(self, action_type, executor):
        self._executors[action_type] = executor
        return executor

    def supports(self, action):
        return type(action) in self._executors or getattr(action, "action_type", None) in self._executors

    def execute(self, action, context):
        executor = self._executors.get(type(action))
        if executor is None:
            executor = self._executors.get(getattr(action, "action_type", None))
        if executor is None:
            raise UnsupportedFeatureError(action)
        return executor(action, context)


def _action_error(action, verb, target, error):
    if isinstance(error, EvaluationError):
        diagnostic = error.diagnostic
        message, code = diagnostic.message, diagnostic.code
    else:
        message, code = str(error), None
    return EvaluationError("Cannot {} '{}': {}".format(verb, target, message), code=code,
                           source=getattr(action, "source", None))


class ConditionExecutorRegistry(ActionExecutorRegistry):
    """Dispatch semantic conditions without coupling them to a CARLA backend."""

    def evaluate(self, condition, context):
        return self.execute(condition, context)


def register_variable_executors(registry):
    """Register backend-independent variable actions on a registry."""
    def set_variable(action, context):
        try:
            return context.evaluation.set_variable(action.variable_ref, action.value)
        except (KeyError, TypeError, ValueError) as error:
            raise _action_error(action, "set variable", action.variable_ref, error) from error

    def modify_variable(action, context):
        try:
            return context.evaluation.modify_variable(action.variable_ref, action.rule, action.value)
        except (KeyError, TypeError, ValueError) as error:
            raise _action_error(action, "modify variable", action.variable_ref, error) from error

    registry.register("VariableSetAction", set_variable)
    registry.register("VariableModifyAction", modify_variable)
    return registry


def register_parameter_executors(registry):
    """Register backend-independent ParameterAction execution on a registry."""
    def parameter_action(action, context):
        try:
            if action.is_modify:
                return context.evaluation.modify_parameter(action.parameter_ref, action.modify_rule,
                                                           action.modify_value)
            return context.evaluation.set_parameter(action.parameter_ref, action.set_value)
        except (KeyError, TypeError, ValueError) as error:
            raise _action_error(action, "update parameter", action.parameter_ref, error) from error

    registry.register("ParameterAction", parameter_action)
    return registry


def _backend_executor(operation):
    def execute(subject, context):
        method = getattr(context.backend, operation, None) if context.backend is not None else None
        if method is None:
            raise backend_capability_error(subject, "Backend does not implement {} for {}".format(
                operation, getattr(subject, "action_type", type(subject).__name__)))
        return method(subject, context)
    return execute


def register_route_executors(registry):
    """Register backend hooks for routes and trajectories."""
    registry.register("AssignRouteAction", _backend_executor("assign_route"))
    registry.register("AcquirePositionAction", _backend_executor("acquire_position"))
    registry.register("FollowTrajectoryAction", _backend_executor("follow_trajectory"))
    return registry


def register_longitudinal_executors(registry):
    """Register backend hooks for longitudinal actions."""
    registry.register("SpeedProfileAction", _backend_executor("apply_speed_profile"))
    return registry


def register_condition_executors(registry):
    """Register condition hooks implemented by the selected backend."""
    registry.register("RelativeAngleCondition", _backend_executor("evaluate_relative_angle"))
    registry.register("RelativeClearanceCondition", _backend_executor("evaluate_relative_clearance"))
    return registry


def default_registry():
    """Registry with every semantic action executor that is migrated off the legacy parser."""
    registry = ActionExecutorRegistry()
    for register in (register_variable_executors, register_parameter_executors, register_route_executors,
                     register_longitudinal_executors):
        register(registry)
    return registry


def default_condition_registry():
    """Registry with every semantic condition executor that is migrated off the legacy parser."""
    return register_condition_executors(ConditionExecutorRegistry())
