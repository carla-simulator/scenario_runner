#!/usr/bin/env python
"""CARLA-free tests for the OpenSCENARIO semantic core."""

from __future__ import absolute_import

import unittest
import xml.etree.ElementTree as ET

from srunner.openscenario.execution.context import ExecutionContext
from srunner.openscenario.execution.registry import ActionExecutorRegistry, register_variable_executors
from srunner.openscenario.execution.registry import (
    ConditionExecutorRegistry, UnsupportedFeatureError, register_condition_executors,
    register_route_executors)
from srunner.openscenario.execution.resolution import resolve_position
from srunner.openscenario.model.catalogs import CatalogResolver
from srunner.openscenario.model.document import OpenScenarioDocument
from srunner.openscenario.model.parameters import (
    EvaluationContext, ParameterDeclaration, ParameterStore, VariableDeclaration, VariableStore)
from srunner.openscenario.model.references import CatalogReference, Position
from srunner.openscenario.model.references import Route, Trajectory
from srunner.openscenario.model.trajectories import TrajectorySamplingError, polyline_samples
from srunner.openscenario.parsing.loader import OpenScenarioLoader
from srunner.openscenario.validation.semantic import SemanticValidator


class TestOpenScenarioSemanticCore(unittest.TestCase):
    def test_typed_parameter_references_are_centralized(self):
        store = ParameterStore([ParameterDeclaration("Speed", "double", "10.5")])
        self.assertEqual(store.resolve("$Speed"), 10.5)
        self.assertEqual(store.resolve("2.0", "double"), 2.0)
        with self.assertRaises(ValueError):
            store.resolve("not-a-number", "double")
        with self.assertRaises(KeyError):
            store.resolve("$Missing")

    def test_variable_actions_use_typed_runtime_store(self):
        xml = ET.fromstring("""<OpenSCENARIO><FileHeader revMajor='1' revMinor='3'/>
          <VariableDeclarations><VariableDeclaration name='count' variableType='int' value='1'/></VariableDeclarations>
          <Storyboard><Init><Actions><GlobalAction><VariableAction variableRef='count'>
            <SetAction value='2'/></VariableAction></GlobalAction>
            <GlobalAction><VariableAction variableRef='count'><ModifyAction><Rule>
              <AddValue value='3'/></Rule></ModifyAction></VariableAction></GlobalAction></Actions></Init></Storyboard>
        </OpenSCENARIO>""")
        document = OpenScenarioLoader().load(xml)
        context = ExecutionContext(state={"evaluation": EvaluationContext(
            parameters=ParameterStore(), variables=document.variable_store)})
        registry = register_variable_executors(ActionExecutorRegistry())
        for action in document.variable_actions():
            registry.execute(action, context)
        self.assertEqual(document.variable_store.get("count"), 5)

    def test_catalog_resolution_retains_semantics_and_assignments(self):
        entry = ET.fromstring("<Vehicle name='car'><ParameterDeclarations/></Vehicle>")
        reference = CatalogReference("VehicleCatalog", "car", {"model": "sedan"})
        resolved = CatalogResolver({"VehicleCatalog": {"car": entry}}).resolve(reference)
        self.assertEqual(resolved.catalog_type, "VehicleCatalog")
        self.assertEqual(resolved.name, "car")
        self.assertEqual(resolved.parameters["model"], "sedan")
        self.assertIsNot(resolved.element, entry)

    def test_positions_are_semantic_before_backend_resolution(self):
        position = Position.from_xml(ET.fromstring(
            "<Position><WorldPosition x='1' y='2' z='0' h='0'/></Position>"))
        self.assertEqual(position.kind, "WorldPosition")
        self.assertEqual(position.attributes["x"], "1")

        route_position = Position.from_xml(ET.fromstring("""
            <Position><RoutePosition><RouteRef><CatalogReference catalogName='Routes' entryName='r1'/>
            </RouteRef><InRoutePosition><FromCurrentEntity entityRef='ego'/></InRoutePosition>
            </RoutePosition></Position>"""))
        self.assertEqual(route_position.kind, "RoutePosition")
        self.assertEqual(route_position.attributes["route_ref"]["entryName"], "r1")

    def test_route_position_validation_is_deterministic(self):
        root = ET.fromstring("""<OpenSCENARIO><FileHeader revMajor='1' revMinor='3'/>
          <Position><RoutePosition><RouteRef/><InRoutePosition/></RoutePosition></Position>
        </OpenSCENARIO>""")
        document = OpenScenarioDocument(root, OpenScenarioLoader().load(
            ET.fromstring("<OpenSCENARIO><FileHeader revMajor='1' revMinor='3'/></OpenSCENARIO>")).version)
        diagnostics = SemanticValidator().validate(document, raise_on_error=False)
        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(diagnostics[0].category, "semantic_error")

    def test_route_and_polyline_trajectory_preserve_semantics(self):
        route = Route.from_xml(ET.fromstring("""
          <Route name='r' closed='false'>
            <Waypoint routeStrategy='shortest'><Position><WorldPosition x='1' y='2' z='0' h='0'/></Position></Waypoint>
            <Waypoint routeStrategy='fastest'><Position><WorldPosition x='3' y='4' z='0' h='0'/></Position></Waypoint>
          </Route>"""))
        self.assertEqual(route.name, "r")
        self.assertEqual([p["route_strategy"] for p in route.positions], ["shortest", "fastest"])
        trajectory = Trajectory.from_xml(ET.fromstring("""
          <Trajectory name='t' closed='false'><Shape><Polyline>
            <Vertex time='0'><Position><WorldPosition x='0' y='0' z='0' h='0'/></Position></Vertex>
            <Vertex time='2'><Position><WorldPosition x='5' y='0' z='0' h='0'/></Position></Vertex>
          </Polyline></Shape></Trajectory>"""),
          {"kind": "Timing", "attributes": {"domainAbsoluteRelative": "absolute"}}, "position")
        self.assertEqual(trajectory.shape, "Polyline")
        self.assertEqual(trajectory.control_points[1]["time"], 2.0)
        self.assertEqual(trajectory.following_mode, "position")

    def test_route_actions_use_backend_boundary(self):
        xml = ET.fromstring("""<OpenSCENARIO><FileHeader revMajor='1' revMinor='3'/>
          <Storyboard><Story><Act><ManeuverGroup><Maneuver><Event><Action>
          <PrivateAction><RoutingAction><AcquirePositionAction><Position>
          <WorldPosition x='1' y='2' z='0' h='0'/></Position></AcquirePositionAction>
          </RoutingAction></PrivateAction></Action></Event></Maneuver></ManeuverGroup></Act></Story></Storyboard>
        </OpenSCENARIO>""")
        document = OpenScenarioLoader().load(xml)
        action = document.routing_actions()[0]
        calls = []
        class Backend(object):
            def acquire_position(self, value, context):
                calls.append(value.position.kind)
                return "resolved"
        registry = register_route_executors(ActionExecutorRegistry())
        result = registry.execute(action, ExecutionContext(backend=Backend()))
        self.assertEqual(result, "resolved")
        self.assertEqual(calls, ["WorldPosition"])

    def test_unsupported_trajectory_is_structured(self):
        action = __import__("srunner.openscenario.model.actions", fromlist=["FollowTrajectoryAction"]).FollowTrajectoryAction()
        registry = register_route_executors(ActionExecutorRegistry())
        with self.assertRaises(UnsupportedFeatureError) as error:
            registry.execute(action, ExecutionContext())
        self.assertEqual(error.exception.diagnostic.category, "unsupported_feature")

    def test_relative_conditions_use_condition_registry(self):
        from srunner.openscenario.model.references import RelativeAngleCondition
        condition = RelativeAngleCondition("npc", "0.2", "lessThan")
        class Backend(object):
            def evaluate_relative_angle(self, value, context):
                return value.entity_ref
        registry = register_condition_executors(ConditionExecutorRegistry())
        self.assertEqual(registry.evaluate(condition, ExecutionContext(backend=Backend())), "npc")

    def test_position_resolution_is_backend_bound(self):
        position = Position.from_xml(ET.fromstring(
            "<Position><RoutePosition><RouteRef><Route name='r' closed='false'>"
            "<Waypoint routeStrategy='shortest'><Position><WorldPosition x='0' y='0' z='0' h='0'/>"
            "</Position></Waypoint><Waypoint routeStrategy='shortest'><Position><WorldPosition x='1' y='0' z='0' h='0'/>"
            "</Position></Waypoint></Route></RouteRef><InRoutePosition><FromRoadCoordinates pathS='0' t='0'/>"
            "</InRoutePosition></RoutePosition></Position>"))
        class Backend(object):
            def resolve_position(self, value, context):
                return value.kind
        self.assertEqual(resolve_position(position, ExecutionContext(backend=Backend())), "RoutePosition")

    def test_polyline_sampling_and_shape_diagnostic(self):
        trajectory = Trajectory.from_xml(ET.fromstring("""
          <Trajectory name='t' closed='false'><Shape><Polyline>
            <Vertex time='0'><Position><WorldPosition x='0' y='0' z='0' h='0'/></Position></Vertex>
            <Vertex time='2'><Position><WorldPosition x='5' y='0' z='0' h='0'/></Position></Vertex>
          </Polyline></Shape></Trajectory>"""))
        self.assertEqual(len(polyline_samples(trajectory)), 2)
        clothoid = Trajectory.from_xml(ET.fromstring("""
          <Trajectory name='c' closed='false'><Shape><Clothoid curvature='0' curvatureDot='0' length='1'>
          <Position><WorldPosition x='0' y='0' z='0' h='0'/></Position></Clothoid></Shape></Trajectory>"""))
        with self.assertRaises(TrajectorySamplingError) as error:
            polyline_samples(clothoid)
        self.assertEqual(error.exception.diagnostic.category, "unsupported_feature")


if __name__ == "__main__":
    unittest.main()
