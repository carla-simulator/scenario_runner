"""Semantic execution dispatch and backend capability boundaries."""

from srunner.openscenario.execution.capabilities import BackendCapabilities
from srunner.openscenario.execution.context import ExecutionContext
from srunner.openscenario.execution.registry import (
    ActionExecutorRegistry, UnsupportedFeatureError, default_registry, register_parameter_executors,
    register_variable_executors)

__all__ = ["BackendCapabilities", "ExecutionContext", "ActionExecutorRegistry", "UnsupportedFeatureError",
           "default_registry", "register_parameter_executors", "register_variable_executors"]
