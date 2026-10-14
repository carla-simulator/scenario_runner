#!/usr/bin/env python
"""Integration tests for routes, trajectories, speed profiles and relative conditions.

Real ``.xosc`` fixtures go through OpenScenarioConfiguration and OpenScenario against the
mocked CARLA module in ``srunner/tests/carla_mocks`` (put it first on PYTHONPATH).
"""

# pylint: disable=protected-access

from __future__ import absolute_import

import math
import os
import shutil
import tempfile
import unittest
import xml.etree.ElementTree as ET
from unittest import mock

import carla
import py_trees

from srunner.openscenario.compatibility.backend import (
    ScenarioRunnerBackend, build_private_action, typed_position_to_transform)
from srunner.openscenario.execution.registry import UnsupportedFeatureError
from srunner.openscenario.model.references import Position
from srunner.openscenario.validation.diagnostics import ValidationError
from srunner.scenarioconfigs.openscenario_configuration import OpenScenarioConfiguration
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (
    ActorTransformSetterToOSCPosition, ChangeActorTargetSpeed, ChangeActorWaypoints)
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (
    RelativeAngleToActor, RelativeClearance, TargetSpeedReached, clearance_occupied, relative_angle)
from srunner.scenariomanager.timer import GameTime
from srunner.scenarios.open_scenario import OpenScenario
from srunner.tools.openscenario_parser import OpenScenarioParser

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(TESTS_DIR, "fixtures", "openscenario")
ROUTING = os.path.join(FIXTURES, "routing_actions.xosc")
PRIVATE_1_3 = os.path.join(FIXTURES, "private_actions_1_3.xosc")
MOCKED_CARLA = os.path.join("carla_mocks", "carla.py") in getattr(carla, "__file__", "")


class _Controller(object):

    def __init__(self):
        self.calls = []

    def update_waypoints(self, waypoints, times=None, start_time=None):
        self.calls.append((waypoints, times, start_time))


@unittest.skipUnless(MOCKED_CARLA, "requires srunner/tests/carla_mocks first on PYTHONPATH")
class TestOpenScenarioRoutingRuntime(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        py_trees.blackboard.Blackboard().__dict__.clear()

    def tearDown(self):
        CarlaDataProvider.cleanup()
        OpenScenarioParser.set_execution_context(None)
        shutil.rmtree(self.tmp)

    @staticmethod
    def _load(path, schema_gate=True):
        """Build the scenario; without ``schema_gate`` the 1.0-only version and XSD checks are bypassed."""
        patches = []
        if not schema_gate:
            patches = [mock.patch.object(OpenScenarioConfiguration, "_check_version", lambda self: None),
                       mock.patch.object(OpenScenarioConfiguration, "_validate_openscenario_configuration",
                                         lambda self: None)]
        for patch in patches:
            patch.start()
        try:
            client = carla.Client()
            config = OpenScenarioConfiguration(path, client, {})
            CarlaDataProvider.set_client(client)
            ego_vehicles = [CarlaDataProvider.request_new_actor(vehicle.model, vehicle.transform, vehicle.rolename,
                                                                color=vehicle.color, actor_category=vehicle.category)
                            for vehicle in config.ego_vehicles]
            return OpenScenario(world=client.get_world(), ego_vehicles=ego_vehicles, config=config,
                                config_file=path, timeout=100000)
        finally:
            for patch in patches:
                patch.stop()

    def _variant(self, source, edit):
        root = ET.parse(source).getroot()
        edit(root)
        path = os.path.join(self.tmp, os.path.basename(source))
        ET.ElementTree(root).write(path)
        return path

    @staticmethod
    def _nodes(scenario):
        return list(scenario.scenario_tree.iterate())

    def _node(self, scenario, name, node_type=None, parent=None):
        matches = [node for node in self._nodes(scenario) if node.name == name and
                   (node_type is None or isinstance(node, node_type)) and
                   (parent is None or (node.parent is not None and node.parent.name == parent))]
        self.assertEqual(len(matches), 1, "expected one node {!r}, got {}".format(name, matches))
        return matches[0]

    @staticmethod
    def _actor(scenario, role_name):
        return next(actor for actor in scenario.other_actors + scenario.ego_vehicles
                    if actor.attributes["role_name"] == role_name)

    def test_shipped_acquire_position_examples_build(self):
        for example in ("IntersectionCollisionAvoidance.xosc", "SyncArrivalIntersection.xosc"):
            scenario = self._load(os.path.join(TESTS_DIR, "..", "examples", example))
            atomic = self._node(scenario, "RouteCreation", ChangeActorWaypoints)
            self.assertIsNone(atomic._times)
            self.assertEqual(atomic._waypoints[0][1], "fastest")
            CarlaDataProvider.cleanup()

    def test_init_teleport_reaches_actor_configuration(self):
        scenario = self._load(ROUTING)
        transforms = dict((actor.rolename, actor.transform) for actor in scenario.config.other_actors)
        self.assertEqual((transforms["npc"].location.x, transforms["npc"].location.y), (5.0, 2.0))
        self.assertEqual((transforms["follower"].location.x, transforms["follower"].location.y), (0.0, 5.0))

    def test_init_assign_route_carries_typed_route(self):
        scenario = self._load(ROUTING)
        atomic = self._node(scenario, "AssignRouteAction", ChangeActorWaypoints, parent="InitActornpc")
        self.assertEqual(atomic.route.name, "NpcRoute")
        self.assertTrue(all(isinstance(position, Position) for position, _ in atomic._waypoints))
        self.assertEqual([strategy for _, strategy in atomic._waypoints], ["shortest", "fastest", "shortest"])
        self.assertIsNone(atomic._times)
        self.assertIs(atomic._position_converter, typed_position_to_transform)
        first = atomic._position_converter(atomic._waypoints[0][0])
        self.assertEqual((first.location.x, first.location.y), (10.0, 2.0))

    def test_init_follow_trajectory_preserves_timing_and_following_mode(self):
        scenario = self._load(ROUTING)
        atomic = self._node(scenario, "FollowTrajectoryAction", ChangeActorWaypoints, parent="InitActorfollower")
        self.assertEqual(atomic.time_reference.kind, "Timing")
        self.assertEqual(atomic.time_reference.domain, "relative")
        self.assertEqual(atomic.following_mode, "position")
        self.assertEqual(atomic._times, [1.0, 5.0, 9.0])
        self.assertFalse(atomic._times_absolute)
        self.assertEqual(atomic.trajectory.name, "InitTrajectory")
        warnings = [d for d in scenario.execution_context.diagnostics if d.category == "approximation_warning"]
        self.assertTrue(any("followingMode 'position'" in d.message for d in warnings))

        follower = self._actor(scenario, "follower")
        controller = _Controller()
        py_trees.blackboard.Blackboard().set("ActorsWithController", {follower.id: controller})
        atomic.initialise()
        route, times, start_time = controller.calls[0]
        self.assertEqual([(t.location.x, t.location.y) for t in route], [(0.0, 5.0), (10.0, 5.0), (20.0, 5.0)])
        self.assertEqual(times, [1.0, 5.0, 9.0])
        self.assertIsNot(times, atomic._times)
        self.assertEqual(start_time, GameTime.get_time())

    def test_absolute_timing_is_relative_to_the_simulation_start(self):
        scenario = self._load(ROUTING)
        element = scenario.config.init.find(".//Private[@entityRef='follower']/PrivateAction/RoutingAction")
        element.find(".//Timing").set("domainAbsoluteRelative", "absolute")
        follower = self._actor(scenario, "follower")
        atomic = build_private_action(element, follower, [follower], {}, scenario.execution_context)
        self.assertTrue(atomic._times_absolute)
        controller = _Controller()
        py_trees.blackboard.Blackboard().set("ActorsWithController", {follower.id: controller})
        with mock.patch.object(GameTime, "_current_game_time", 3.0):
            atomic.initialise()
        self.assertEqual(controller.calls[0][1], [-2.0, 2.0, 6.0])

    def test_story_level_routing_actions_use_semantic_runtime(self):
        scenario = self._load(ROUTING)
        acquire = self._node(scenario, "StoryAcquirePosition", ChangeActorWaypoints)
        self.assertEqual(acquire.position.kind, "WorldPosition")
        self.assertEqual(acquire._waypoints[0][1], "fastest")
        route = self._node(scenario, "StoryAssignRoute", ChangeActorWaypoints)
        self.assertEqual(route.route.name, "NpcRoute")
        trajectory = self._node(scenario, "StoryFollowTrajectory", ChangeActorWaypoints)
        self.assertEqual(trajectory.time_reference.kind, "None")
        self.assertEqual(trajectory.following_mode, "follow")
        self.assertIsNone(trajectory._times)
        self.assertEqual(len(trajectory._waypoints), 3)

    def test_unmigrated_action_falls_back_to_legacy(self):
        scenario = self._load(ROUTING)
        self._node(scenario, "LegacyTeleport", ActorTransformSetterToOSCPosition)
        teleport = scenario.config.xml_tree.find(".//Event[@name='LegacyTeleport']//PrivateAction")
        with self.assertRaises(UnsupportedFeatureError) as raised:
            build_private_action(teleport, None, [], {}, scenario.execution_context)
        self.assertTrue(raised.exception.not_migrated)

    def test_programming_errors_are_not_hidden_by_the_fallback(self):
        with mock.patch.object(ScenarioRunnerBackend, "assign_route", side_effect=AttributeError("bug")):
            with self.assertRaises(AttributeError):
                self._load(ROUTING)

    def test_one_vertex_polyline_is_a_validation_error(self):
        def edit(root):
            polyline = root.find(".//Trajectory[@name='InitTrajectory']//Polyline")
            for vertex in polyline.findall("Vertex")[1:]:
                polyline.remove(vertex)
        with self.assertRaises(ValidationError) as raised:
            self._load(self._variant(ROUTING, edit), schema_gate=False)
        self.assertEqual(raised.exception.diagnostics[0].category, "semantic_error")
        self.assertIn("at least two vertices", raised.exception.diagnostics[0].message)

    def test_unsupported_trajectory_shapes_are_backend_diagnostics(self):
        clothoid = ET.fromstring(
            '<Shape><Clothoid curvature="0.01" curvatureDot="0" length="20">'
            '<Position><WorldPosition x="0" y="5" z="0" h="0"/></Position></Clothoid></Shape>')
        for trajectory in ("InitTrajectory", "StoryTrajectory"):
            def edit(root, name=trajectory):
                element = root.find(".//Trajectory[@name='{}']".format(name))
                element.remove(element.find("Shape"))
                element.append(clothoid)
            with self.assertRaises(UnsupportedFeatureError) as raised:
                self._load(self._variant(ROUTING, edit))
            self.assertEqual(raised.exception.diagnostic.category, "backend_capability_error")
            self.assertIn("Clothoid", raised.exception.diagnostic.message)

    def test_route_position_is_a_backend_diagnostic(self):
        def edit(root):
            position = root.find(".//Event[@name='StoryAcquirePosition']//AcquirePositionAction/Position")
            position.remove(position[0])
            position.append(ET.fromstring(
                '<RoutePosition><RouteRef><CatalogReference catalogName="Routes" entryName="r"/></RouteRef>'
                '<InRoutePosition><FromCurrentEntity entityRef="npc"/></InRoutePosition></RoutePosition>'))
        with self.assertRaises(UnsupportedFeatureError) as raised:
            self._load(self._variant(ROUTING, edit))
        self.assertEqual(raised.exception.diagnostic.category, "backend_capability_error")
        self.assertIn("RoutePosition", raised.exception.diagnostic.message)

    def test_modify_action_without_rule_is_reported(self):
        def edit(root):
            actions = root.find("Storyboard/Init/Actions")
            actions.insert(0, ET.fromstring(
                '<GlobalAction><ParameterAction parameterRef="TimeScale"><ModifyAction/></ParameterAction>'
                '</GlobalAction>'))
        with self.assertRaises(ValidationError) as raised:
            self._load(self._variant(ROUTING, edit), schema_gate=False)
        self.assertIn("requires a Rule", raised.exception.diagnostics[0].message)

    def test_speed_profile_actions_build_behaviours(self):
        scenario = self._load(PRIVATE_1_3, schema_gate=False)
        init_profile = self._node(scenario, "SpeedProfileAction", parent="InitActornpc")
        self.assertEqual(init_profile.entries, [(5.0, 2.0), (8.0, None)])
        timed, final = init_profile.children
        self.assertIsInstance(timed, ChangeActorTargetSpeed)
        self.assertEqual((timed._target_speed, timed._duration), (5.0, 2.0))
        self.assertIsInstance(final.children[0], ChangeActorTargetSpeed)
        self.assertTrue(final.children[0]._continuous)
        self.assertIsInstance(final.children[1], TargetSpeedReached)
        self.assertEqual(final.children[1]._target_speed, 8.0)

        story_profile = self._node(scenario, "StorySpeedProfile", py_trees.composites.Sequence,
                                   parent="StorySpeedProfile")
        relative = story_profile.children[0]
        self.assertIs(relative._relative_actor, self._actor(scenario, "hero"))
        self.assertEqual((relative._value, relative._value_type), (5.0, "delta"))

    def test_trajectory_ref_is_followed(self):
        scenario = self._load(PRIVATE_1_3, schema_gate=False)
        atomic = self._node(scenario, "FollowTrajectoryAction", ChangeActorWaypoints, parent="InitActorfollower")
        self.assertEqual(atomic.trajectory.name, "InitTrajectory")
        self.assertEqual(atomic._times, [1.0, 5.0, 9.0])

    def test_relative_conditions_build_condition_nodes(self):
        scenario = self._load(PRIVATE_1_3, schema_gate=False)
        hero = self._actor(scenario, "hero")
        angle = self._node(scenario, "AngleStart", RelativeAngleToActor)
        self.assertIs(angle._reference_actor, hero)
        self.assertEqual((angle._angle_type, angle._angle, angle._tolerance), ("heading", 0.0, 0.1))
        clearance = self._node(scenario, "ClearanceStart", RelativeClearance)
        self.assertEqual(clearance._other_actors, [hero])
        self.assertEqual(clearance._lane_ranges, [(-1, 1)])
        self.assertEqual((clearance._distance_forward, clearance._distance_backward), (20.0, 5.0))
        self.assertIs(clearance._freespace, True)
        self.assertIs(clearance._opposite_lanes, False)

    def test_unsupported_angle_coordinate_system_is_a_backend_diagnostic(self):
        def edit(root):
            root.find(".//RelativeAngleCondition").set("coordinateSystem", "lane")
        with self.assertRaises(UnsupportedFeatureError) as raised:
            self._load(self._variant(PRIVATE_1_3, edit), schema_gate=False)
        self.assertEqual(raised.exception.diagnostic.category, "backend_capability_error")

    def test_unknown_entity_condition_is_reported_not_returned_as_none(self):
        condition = ET.fromstring(
            '<Condition name="Angle" delay="0" conditionEdge="rising"><ByEntityCondition>'
            '<TriggeringEntities triggeringEntitiesRule="any"><EntityRef entityRef="npc"/></TriggeringEntities>'
            '<EntityCondition><AngleCondition angleType="heading" angle="0" angleTolerance="0.1"/></EntityCondition>'
            '</ByEntityCondition></Condition>')
        with self.assertRaises(UnsupportedFeatureError) as raised:
            OpenScenarioParser.convert_condition_to_atomic(condition, [None])
        self.assertIn("Angle", raised.exception.diagnostic.message)


class TestRelativeConditionGeometry(unittest.TestCase):

    def test_relative_heading_uses_the_openscenario_convention(self):
        rotation = carla.Rotation(yaw=100)
        reference = carla.Rotation(yaw=90)
        self.assertAlmostEqual(relative_angle(rotation, reference, "heading"), -math.radians(10))
        self.assertAlmostEqual(relative_angle(rotation, reference, "heading", right_handed=False), math.radians(10))
        self.assertAlmostEqual(relative_angle(carla.Rotation(yaw=-170), carla.Rotation(yaw=170), "heading",
                                              right_handed=False), math.radians(20))

    def test_clearance_area(self):
        self.assertTrue(clearance_occupied(10.0, 0, False, 20.0, 5.0))
        self.assertFalse(clearance_occupied(25.0, 0, False, 20.0, 5.0))
        self.assertTrue(clearance_occupied(-5.0, 0, False, 20.0, 5.0))
        self.assertFalse(clearance_occupied(-6.0, 0, False, 20.0, 5.0))
        self.assertFalse(clearance_occupied(1.0, 2, False, 20.0, 5.0, [(-1, 1)]))
        self.assertTrue(clearance_occupied(1.0, 2, False, 20.0, 5.0, [(-1, 1), (2, None)]))
        self.assertFalse(clearance_occupied(1.0, 1, True, 20.0, 5.0))
        self.assertTrue(clearance_occupied(1.0, 1, True, 20.0, 5.0, opposite_lanes=True))


if __name__ == "__main__":
    unittest.main()
