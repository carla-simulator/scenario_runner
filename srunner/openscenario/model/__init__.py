"""Backend-independent OpenSCENARIO semantic model."""

from srunner.openscenario.model.document import OpenScenarioDocument
from srunner.openscenario.model.parameters import (
    EvaluationContext, ParameterDeclaration, ParameterScope, ParameterStore, ParameterValue,
    VariableDeclaration, VariableStore)
from srunner.openscenario.model.actions import (
    AcquirePositionAction, AssignRouteAction, FollowTrajectoryAction, ParameterAction,
    SpeedProfileAction, VariableModifyAction, VariableSetAction)
from srunner.openscenario.model.references import (
    CatalogEntry, CatalogReference, Condition, EntityReference, Position, RelativeAngleCondition,
    RelativeClearanceCondition, Route, RoutePosition, Trajectory)
from srunner.openscenario.model.catalogs import CatalogResolver
from srunner.openscenario.model.trajectories import (
    TrajectorySamplingError, backend_sampling_required, polyline_samples)

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
    "AssignRouteAction",
    "AcquirePositionAction",
    "FollowTrajectoryAction",
    "SpeedProfileAction",
    "CatalogEntry",
    "CatalogResolver",
    "CatalogReference",
    "Condition",
    "EntityReference",
    "Position",
    "Route",
    "RoutePosition",
    "Trajectory",
    "RelativeAngleCondition",
    "RelativeClearanceCondition",
    "TrajectorySamplingError",
    "polyline_samples",
    "backend_sampling_required",
]
