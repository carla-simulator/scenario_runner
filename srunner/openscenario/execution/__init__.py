"""Semantic execution dispatch and backend capability boundaries."""

from srunner.openscenario.execution.capabilities import BackendCapabilities
from srunner.openscenario.execution.context import ExecutionContext
from srunner.openscenario.execution.registry import (
    ActionExecutorRegistry, ConditionExecutorRegistry, UnsupportedFeatureError, default_registry,
    register_condition_executors, register_parameter_executors, register_route_executors,
    register_variable_executors)
from srunner.openscenario.execution.resolution import resolve_position

__all__ = ["BackendCapabilities", "ExecutionContext", "ActionExecutorRegistry", "ConditionExecutorRegistry",
           "UnsupportedFeatureError", "default_registry", "register_condition_executors",
           "register_parameter_executors", "register_route_executors", "register_variable_executors",
           "resolve_position"]
