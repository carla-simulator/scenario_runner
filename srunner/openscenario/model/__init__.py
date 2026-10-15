"""Backend-independent OpenSCENARIO semantic model."""

from srunner.openscenario.model.document import OpenScenarioDocument
from srunner.openscenario.model.parameters import (
    EvaluationContext, ParameterDeclaration, ParameterScope, ParameterStore, ParameterValue,
    VariableDeclaration, VariableStore)
from srunner.openscenario.model.actions import ParameterAction, VariableModifyAction, VariableSetAction
from srunner.openscenario.model.references import (
    CatalogEntry, CatalogReference, Condition, EntityReference, Position, Route, RoutePosition, Trajectory)
from srunner.openscenario.model.catalogs import CatalogResolver

__all__ = [
    "OpenScenarioDocument",
    "ParameterDeclaration",
    "ParameterScope",
    "VariableDeclaration",
    "ParameterValue",
    "ParameterStore",
    "VariableStore",
    "EvaluationContext",
    "ParameterAction",
    "VariableSetAction",
    "VariableModifyAction",
    "CatalogEntry",
    "CatalogResolver",
    "CatalogReference",
    "Condition",
    "EntityReference",
    "Position",
    "Route",
    "RoutePosition",
    "Trajectory",
]
