"""Registry-based dispatch for semantic actions."""

from __future__ import absolute_import

from srunner.openscenario.model.expressions import EvaluationError
from srunner.openscenario.validation.diagnostics import Diagnostic


class UnsupportedFeatureError(RuntimeError):
    def __init__(self, action, diagnostic=None):
        self.action = action
        self.diagnostic = diagnostic or Diagnostic(
            "unsupported_feature", "No executor registered for {}".format(type(action).__name__))
        super(UnsupportedFeatureError, self).__init__(str(self.diagnostic))


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


def default_registry():
    """Registry with every semantic executor that is migrated off the legacy parser."""
    return register_parameter_executors(register_variable_executors(ActionExecutorRegistry()))


def register_route_executors(registry):
    """Register backend hooks for routes and trajectories."""
    def backend_operation(action, context, operation):
        method = getattr(context.backend, operation, None) if context.backend is not None else None
        if method is None:
            raise UnsupportedFeatureError(action, Diagnostic(
                "unsupported_feature", "Backend does not implement {}".format(operation),
                source=getattr(action, "source", None)))
        return method(action, context)

    registry.register("AssignRouteAction", lambda action, context: backend_operation(
        action, context, "assign_route"))
    registry.register("AcquirePositionAction", lambda action, context: backend_operation(
        action, context, "acquire_position"))
    registry.register("FollowTrajectoryAction", lambda action, context: backend_operation(
        action, context, "follow_trajectory"))
    registry.register("SpeedProfileAction", lambda action, context: backend_operation(
        action, context, "apply_speed_profile"))
    return registry


def register_condition_executors(registry):
    """Register condition hooks implemented by the selected backend."""
    def backend_condition(condition, context, operation):
        method = getattr(context.backend, operation, None) if context.backend is not None else None
        if method is None:
            raise UnsupportedFeatureError(condition, Diagnostic(
                "unsupported_feature", "Backend does not implement {}".format(operation),
                source=getattr(condition, "source", None)))
        return method(condition, context)

    registry.register("RelativeAngleCondition", lambda condition, context: backend_condition(
        condition, context, "evaluate_relative_angle"))
    registry.register("RelativeClearanceCondition", lambda condition, context: backend_condition(
        condition, context, "evaluate_relative_clearance"))
    return registry
