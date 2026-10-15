"""ScenarioRunner backend that turns typed OpenSCENARIO actions and conditions into py_trees behaviours."""

from __future__ import absolute_import

import py_trees

from srunner.openscenario.compatibility.attributes import resolve_attribute
from srunner.openscenario.execution.registry import (
    UnsupportedFeatureError, backend_capability_error, default_condition_registry, default_registry)
from srunner.openscenario.model.conditions import ANGLE_TYPES
from srunner.openscenario.model.document import OpenScenarioDocument
from srunner.openscenario.model.references import FOLLOWING_MODES, Route, TimeReference, Trajectory, invalid
from srunner.openscenario.model.trajectories import timed_samples
from srunner.openscenario.validation.diagnostics import Diagnostic
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import ChangeActorTargetSpeed, ChangeActorWaypoints
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (
    RelativeAngleToActor, RelativeClearance, TargetSpeedReached)

SUPPORTED_POSITIONS = frozenset((
    "WorldPosition", "RelativeWorldPosition", "RelativeObjectPosition", "RelativeLanePosition",
    "RelativeRoadPosition", "RoadPosition", "LanePosition"))
SUPPORTED_ANGLE_COORDINATE_SYSTEMS = ("entity", "world")


def _legacy_parser():
    # Deferred: the legacy parser imports this module to route migrated actions here.
    from srunner.tools.openscenario_parser import OpenScenarioParser  # pylint: disable=import-outside-toplevel
    return OpenScenarioParser


def typed_position_to_transform(position):
    """Convert a typed Position into a carla.Transform with the legacy XML position conversion."""
    return _legacy_parser().convert_position_to_transform(position.source)


def not_migrated(element):
    """UnsupportedFeatureError telling the caller to use the legacy runtime for ``element``."""
    return UnsupportedFeatureError(element, Diagnostic(
        "unsupported_feature", "{} is not migrated to the semantic runtime".format(getattr(element, "tag", element)),
        source=element, code=Diagnostic.NOT_MIGRATED))


class ScenarioRunnerBackend(object):
    """Backend operations invoked by the semantic registries for one actor."""

    def __init__(self, actor=None, actor_list=None, catalogs=None, name=None):
        self.actor = actor
        self.actor_list = [item for item in (actor_list or []) if item is not None]
        self.catalogs = catalogs or {}
        self.name = name

    def _name(self, subject):
        return self.name or subject.action_type

    @staticmethod
    def _resolve(context, value, value_type="double"):
        return resolve_attribute(value, value_type, context.evaluation)

    def _entity(self, entity_ref, source, context):
        entity_ref = self._resolve(context, entity_ref, "string")
        for actor in self.actor_list:
            if actor.attributes.get("role_name") == entity_ref:
                return actor
        raise invalid("Unknown entity reference {!r}".format(entity_ref), source)

    def _catalog_entry(self, reference, tag):
        try:
            entry = _legacy_parser().get_catalog_entry(self.catalogs, reference.source)
        except KeyError as error:
            raise invalid("Unknown catalog entry {}/{}".format(reference.catalog_name, reference.entry_name),
                          reference.source) from error
        if entry.tag != tag:
            raise invalid("Catalog entry {}/{} is a {}, expected a {}".format(
                reference.catalog_name, reference.entry_name, entry.tag, tag), reference.source)
        return entry

    def _following_mode(self, context, action):
        mode = self._resolve(context, action.following_mode, "string")
        if mode not in FOLLOWING_MODES:
            raise invalid("followingMode must be 'follow' or 'position', got {!r}".format(mode), action.source)
        return mode

    @staticmethod
    def _supported_position(position):
        if position.kind not in SUPPORTED_POSITIONS:
            raise backend_capability_error(position, "{} is not supported by the ScenarioRunner backend".format(
                position.kind))
        return position

    def resolve_position(self, position, context):
        return typed_position_to_transform(self._supported_position(position))

    def assign_route(self, action, context):
        route = action.route
        if route is None:
            route = Route.from_xml(self._catalog_entry(action.catalog_reference, "Route"))
        if self._resolve(context, route.closed, "boolean"):
            context.warn(Diagnostic("approximation_warning", "Closed route {!r} is driven once".format(route.name),
                                    severity="warning", source=route.source))
        waypoints = [(self._supported_position(point["position"]),
                      self._resolve(context, point["route_strategy"], "string")) for point in route.positions]
        atomic = ChangeActorWaypoints(self.actor, waypoints=waypoints, name=self._name(action),
                                      position_converter=typed_position_to_transform)
        atomic.route = route
        return atomic

    def acquire_position(self, action, context):
        atomic = ChangeActorWaypoints(self.actor, waypoints=[(self._supported_position(action.position), "fastest")],
                                      name=self._name(action), position_converter=typed_position_to_transform)
        atomic.position = action.position
        return atomic

    def follow_trajectory(self, action, context):
        trajectory = action.trajectory
        if trajectory is None:
            trajectory = Trajectory.from_xml(self._catalog_entry(action.catalog_reference, "Trajectory"))
        if trajectory.shape != "Polyline":
            raise backend_capability_error(trajectory, "{} trajectories are not supported by the ScenarioRunner "
                                           "backend; use a Polyline".format(trajectory.shape))
        if self._resolve(context, trajectory.closed, "boolean"):
            raise backend_capability_error(trajectory, "Closed trajectories are not supported by the "
                                           "ScenarioRunner backend")
        if action.initial_distance_offset is not None and self._resolve(context, action.initial_distance_offset):
            raise backend_capability_error(action, "FollowTrajectoryAction initialDistanceOffset is not supported "
                                           "by the ScenarioRunner backend")

        samples, times = timed_samples(trajectory, action.time_reference,
                                       resolve=lambda value: self._resolve(context, value))
        absolute = False
        if times is not None:
            domain = self._resolve(context, action.time_reference.domain, "string")
            if domain not in TimeReference.DOMAINS:
                raise invalid("Timing domainAbsoluteRelative must be 'absolute' or 'relative', got {!r}".format(
                    domain), action.time_reference.source)
            absolute = domain == "absolute"
        following_mode = self._following_mode(context, action)
        if following_mode == "position":
            context.warn(Diagnostic(
                "approximation_warning", "followingMode 'position' is executed by the actor's controller, which "
                "may not track the trajectory exactly", severity="warning", source=action.source))

        waypoints = [(self._supported_position(sample["position"]), "shortest") for sample in samples]
        atomic = ChangeActorWaypoints(self.actor, waypoints=waypoints, times=times, times_absolute=absolute,
                                      name=self._name(action), position_converter=typed_position_to_transform)
        atomic.trajectory = trajectory
        atomic.time_reference = action.time_reference
        atomic.following_mode = following_mode
        return atomic

    def apply_speed_profile(self, action, context):
        relative_actor = self._entity(action.entity_ref, action.source, context) if action.entity_ref else None
        following_mode = self._following_mode(context, action)
        entries = [(self._resolve(context, speed), self._resolve(context, time) if time is not None else None)
                   for speed, time in action.entries]
        if action.dynamic_constraints or following_mode == "position" or any(t is not None for _, t in entries):
            context.warn(Diagnostic(
                "approximation_warning", "SpeedProfileAction commands each target speed to the actor's controller; "
                "entry times bound each step and DynamicConstraints are not enforced",
                severity="warning", source=action.source))

        profile = py_trees.composites.Sequence(name=self._name(action))
        for index, (speed, time) in enumerate(entries):
            name = "{}_{}".format(self._name(action), index)
            last = index == len(entries) - 1
            if time is not None and not last:
                profile.add_child(self._target_speed(speed, relative_actor, name, duration=time))
                continue
            step = py_trees.composites.Parallel(name=name, policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
            # An infinite duration keeps the target commanded until TargetSpeedReached ends the step.
            step.add_child(self._target_speed(speed, relative_actor, name, duration=float("inf")))
            step.add_child(TargetSpeedReached(self.actor, speed, relative_actor, name=name + "_Reached"))
            profile.add_child(step)
        profile.entries = entries
        profile.following_mode = following_mode
        return profile

    def _target_speed(self, speed, relative_actor, name, duration):
        if relative_actor is not None:
            return ChangeActorTargetSpeed(self.actor, 0.0, duration=duration, relative_actor=relative_actor,
                                          value=speed, value_type="delta", name=name)
        return ChangeActorTargetSpeed(self.actor, speed, duration=duration, name=name)

    def evaluate_relative_angle(self, condition, context):
        angle_type = self._resolve(context, condition.angle_type, "string")
        coordinate_system = self._resolve(context, condition.coordinate_system, "string")
        if angle_type not in ANGLE_TYPES:
            raise invalid("RelativeAngleCondition angleType must be one of {}, got {!r}".format(
                ", ".join(ANGLE_TYPES), angle_type), condition.source)
        if coordinate_system not in SUPPORTED_ANGLE_COORDINATE_SYSTEMS:
            raise backend_capability_error(condition, "RelativeAngleCondition coordinateSystem {!r} is not "
                                           "supported by the ScenarioRunner backend".format(coordinate_system))
        return RelativeAngleToActor(
            self.actor, self._entity(condition.entity_ref, condition.source, context), angle_type,
            self._resolve(context, condition.angle), self._resolve(context, condition.angle_tolerance),
            right_handed=not _legacy_parser().use_carla_coordinate_system, name=self.name or condition.kind)

    def evaluate_relative_clearance(self, condition, context):
        def bound(value):
            return self._resolve(context, value, "int") if value is not None else None

        def distance(value):
            return self._resolve(context, value) if value is not None else 0.0

        others = [self._entity(ref, condition.source, context) for ref in condition.entity_refs] or None
        return RelativeClearance(
            self.actor, others, [(bound(low), bound(high)) for low, high in condition.lane_ranges],
            distance(condition.distance_forward), distance(condition.distance_backward),
            self._resolve(context, condition.freespace, "boolean"),
            self._resolve(context, condition.opposite_lanes, "boolean"), name=self.name or condition.kind)


def build_private_action(element, actor, actor_list, catalogs, context, name=None, registry=None):
    """
    Build the behaviour of a migrated private action through the semantic registry.

    Raises UnsupportedFeatureError whose ``not_migrated`` is True when the legacy runtime must
    handle ``element``; any other UnsupportedFeatureError or ValidationError is final.
    """
    if context is None:
        raise not_migrated(element)
    action = OpenScenarioDocument.parse_private_action(element)
    if action is None:
        raise not_migrated(element)
    backend = ScenarioRunnerBackend(actor, actor_list, catalogs, name)
    return (registry or default_registry()).execute(action, context.with_backend(backend))


def build_entity_condition(element, trigger_actor, actor_list, context, name=None, registry=None):
    """Build the behaviour of a migrated entity condition; see build_private_action for errors."""
    if context is None:
        raise not_migrated(element)
    condition = OpenScenarioDocument.parse_entity_condition(element, name)
    if condition is None:
        raise not_migrated(element)
    backend = ScenarioRunnerBackend(trigger_actor, actor_list, name=name)
    return (registry or default_condition_registry()).evaluate(condition, context.with_backend(backend))
