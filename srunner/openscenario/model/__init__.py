"""Backend-independent OpenSCENARIO semantic model."""

from srunner.openscenario.model.document import OpenScenarioDocument
from srunner.openscenario.model.parameters import ParameterDeclaration, ParameterScope, VariableDeclaration
from srunner.openscenario.model.actions import ParameterAction
from srunner.openscenario.model.references import (
    CatalogReference, Condition, EntityReference, Position, Route, Trajectory)

__all__ = [
    "OpenScenarioDocument",
    "ParameterDeclaration",
    "ParameterScope",
    "VariableDeclaration",
    "ParameterAction",
    "CatalogReference",
    "Condition",
    "EntityReference",
    "Position",
    "Route",
    "Trajectory",
]
