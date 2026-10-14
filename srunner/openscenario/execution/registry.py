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
