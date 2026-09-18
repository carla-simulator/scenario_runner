"""Registry-based dispatch for semantic actions."""

from __future__ import absolute_import

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


def register_variable_executors(registry):
    """Register backend-independent variable actions on a registry."""
    def set_variable(action, context):
        try:
            return context.evaluation.variables.set(action.variable_ref, context.evaluation.resolve(action.value))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Cannot set variable '{}': {}".format(action.variable_ref, error))

    def modify_variable(action, context):
        try:
            value = context.evaluation.resolve(action.value)
            return context.evaluation.variables.modify(action.variable_ref, action.rule, value)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Cannot modify variable '{}': {}".format(action.variable_ref, error))

    registry.register("VariableSetAction", set_variable)
    registry.register("VariableModifyAction", modify_variable)
    return registry
