#!/usr/bin/env python
"""CARLA-free tests for the OpenSCENARIO semantic core."""

from __future__ import absolute_import

import unittest
import xml.etree.ElementTree as ET

from srunner.openscenario.execution.context import ExecutionContext
from srunner.openscenario.execution.registry import ActionExecutorRegistry, register_variable_executors
from srunner.openscenario.model.catalogs import CatalogResolver
from srunner.openscenario.model.document import OpenScenarioDocument
from srunner.openscenario.model.parameters import (
    EvaluationContext, ParameterDeclaration, ParameterStore, VariableDeclaration, VariableStore)
from srunner.openscenario.model.references import CatalogReference, Position
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
          <Storyboard><Init><Actions><GlobalAction><VariableAction>
            <VariableSetAction variableRef='count' value='2'/></VariableAction></GlobalAction>
            <GlobalAction><VariableAction><VariableModifyAction variableRef='count'
              modification='add' value='3'/></VariableAction></GlobalAction></Actions></Init></Storyboard>
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


if __name__ == "__main__":
    unittest.main()
