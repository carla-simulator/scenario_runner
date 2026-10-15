#!/usr/bin/env python
"""Parameter references and expressions in arbitrary OpenSCENARIO attributes.

The end-to-end tests drive ``OpenScenarioConfiguration`` and ``OpenScenario``
and need the CARLA mocks (``srunner/tests/carla_mocks`` on PYTHONPATH); the
remaining tests also run against the real ``carla`` wheel.
"""

from __future__ import absolute_import

import math
import os
import shutil
import tempfile
import unittest
import xml.etree.ElementTree as ET
from types import SimpleNamespace
from unittest import mock

try:
    import carla
    import py_trees
    from srunner.scenarioconfigs.openscenario_configuration import OpenScenarioConfiguration
    from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
    from srunner.scenariomanager.timer import TimeOut
    from srunner.scenarios.open_scenario import OpenScenario, create_execution_context
    from srunner.tools.openscenario_parser import OpenScenarioParser, ParameterRef
    RUNTIME_IMPORT_ERROR = None
    USING_CARLA_MOCKS = "carla_mocks" in (getattr(carla, "__file__", None) or "")
except ImportError as error:  # pragma: no cover - depends on the local CARLA installation
    RUNTIME_IMPORT_ERROR = error
    USING_CARLA_MOCKS = False

from srunner.openscenario.model.expressions import EvaluationError
from srunner.openscenario.parsing.loader import OpenScenarioLoader
from srunner.openscenario.validation.diagnostics import (
    Diagnostic, DiagnosticError, ValidationError, diagnostics_as_validation_error)
from srunner.openscenario.validation.semantic import SemanticValidator

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_SCENARIO = os.path.join(TESTS_DIR, "..", "examples", "FollowLeadingVehicle.xosc")

PARAMETERS = (
    ("Model", "string", "vehicle.tesla.model3"),
    ("Delay", "double", "2.5"),
    ("Override", "boolean", "false"),
    ("Gap", "double", "25"),
    ("Zero", "double", "0"),
)


def scenario_root(parameters=PARAMETERS):
    root = ET.parse(BASE_SCENARIO).getroot()
    declarations = root.find("ParameterDeclarations")
    for name, parameter_type, value in parameters:
        ET.SubElement(declarations, "ParameterDeclaration",
                      {"name": name, "parameterType": parameter_type, "value": value})
    return root


def adversary_vehicle(root):
    return root.find(".//ScenarioObject[@name='adversary']/Vehicle")


def unknown_codes(diagnostics):
    return [item.code for item in diagnostics if item.code == Diagnostic.UNKNOWN_REFERENCE]


@unittest.skipIf(RUNTIME_IMPORT_ERROR is not None or not USING_CARLA_MOCKS,
                 "needs the ScenarioRunner runtime with the CARLA mocks")
class TestAttributeResolutionEntryPoints(unittest.TestCase):

    def setUp(self):
        self._directory = tempfile.mkdtemp()
        self._patches = [
            mock.patch.object(CarlaDataProvider, "_global_osc_parameters", {}),
            # ${...} is not valid against the bundled 1.0 XSD.
            mock.patch.object(OpenScenarioConfiguration, "_validate_openscenario_configuration"),
        ]
        for patch in self._patches:
            patch.start()
        py_trees.blackboard.Blackboard().__dict__.clear()

    def tearDown(self):
        OpenScenarioParser.set_execution_context(None)
        CarlaDataProvider.cleanup()
        for patch in reversed(self._patches):
            patch.stop()
        shutil.rmtree(self._directory)

    def _configuration(self, root, overrides=None):
        path = os.path.join(self._directory, "scenario.xosc")
        ET.ElementTree(root).write(path)
        return OpenScenarioConfiguration(path, carla.Client(), overrides or {})

    def _scenario(self, root, overrides=None):
        config = self._configuration(root, overrides)
        client = carla.Client()
        CarlaDataProvider.set_client(client)
        egos = [CarlaDataProvider.request_new_actor(vehicle.model, vehicle.transform, vehicle.rolename,
                                                    actor_category=vehicle.category)
                for vehicle in config.ego_vehicles]
        scenario = OpenScenario(world=client.get_world(), ego_vehicles=egos, config=config,
                                config_file=config.filename, timeout=1000)
        return config, scenario

    def test_attribute_references_and_expressions_resolve_end_to_end(self):
        root = scenario_root()
        adversary_vehicle(root).set("name", "$Model")
        root.find(".//Sun").set("azimuth", "${1 + 1}")
        # Animation would recompute the sun position from dateTime.
        root.find(".//TimeOfDay").set("animation", "$Override")
        root.find(".//RelativeRoadPosition").set("ds", "${$Gap * 2}")
        root.find(".//Throttle").set("active", "$Override")
        root.find(".//Condition[@name='StartTime']").set("delay", "$Delay")

        config, scenario = self._scenario(root)
        self.assertIsNotNone(scenario)
        self.assertEqual([actor.model for actor in config.other_actors], ["vehicle.tesla.model3"])

        weather = OpenScenarioParser.get_weather_from_env_action(config.init, config.catalogs)
        self.assertAlmostEqual(weather.carla_weather.sun_azimuth_angle, math.degrees(2.0))
        self.assertIs(weather.animation, False)

        condition = OpenScenarioParser.convert_condition_to_atomic(
            config.storyboard.find(".//Condition[@name='StartTime']"), scenario.ego_vehicles)
        delays = [node for node in condition.iterate() if isinstance(node, TimeOut)]
        self.assertEqual([node._timeout_value for node in delays], [2.5])  # pylint: disable=protected-access

    def test_entity_model_uses_parameter_override(self):
        root = scenario_root()
        adversary_vehicle(root).set("name", "$Model")
        config = self._configuration(root, {"Model": "vehicle.audi.tt"})
        self.assertEqual([actor.model for actor in config.other_actors], ["vehicle.audi.tt"])

    def test_undeclared_reference_in_any_attribute_is_a_validation_error(self):
        root = scenario_root()
        root.find(".//RelativeRoadPosition").set("ds", "$nope")
        with self.assertRaises(ValidationError) as raised:
            self._configuration(root)
        diagnostics = raised.exception.diagnostics
        self.assertEqual([item.code for item in diagnostics], [Diagnostic.UNKNOWN_REFERENCE])
        self.assertIn("nope", diagnostics[0].message)
        self.assertEqual(diagnostics[0].source.tag, "RelativeRoadPosition")

    def test_undeclared_entity_model_is_not_replaced_by_a_fallback(self):
        root = scenario_root()
        adversary_vehicle(root).set("name", "$Missing")
        with self.assertRaises(ValidationError) as raised:
            self._configuration(root)
        self.assertEqual(unknown_codes(raised.exception.diagnostics), [Diagnostic.UNKNOWN_REFERENCE])

    def test_missing_file_header_is_a_validation_error(self):
        root = scenario_root()
        root.remove(root.find("FileHeader"))
        with self.assertRaises(ValidationError) as raised:
            self._configuration(root)
        self.assertEqual([item.code for item in raised.exception.diagnostics], [Diagnostic.MALFORMED_DOCUMENT])

    def test_division_by_zero_in_an_attribute_is_a_structured_error(self):
        root = scenario_root()
        root.find(".//Sun").set("azimuth", "${$Gap / $Zero}")
        with self.assertRaises(ValidationError) as raised:
            self._scenario(root)
        self.assertEqual([item.code for item in raised.exception.diagnostics], [Diagnostic.ARITHMETIC_ERROR])
        self.assertIsNone(OpenScenarioParser.execution_context)


@unittest.skipIf(RUNTIME_IMPORT_ERROR is not None, "ScenarioRunner runtime not importable: {}".format(
    RUNTIME_IMPORT_ERROR))
class TestParserAttributeResolution(unittest.TestCase):

    def setUp(self):
        self._patch = mock.patch.object(CarlaDataProvider, "_global_osc_parameters", {})
        self._patch.start()

    def tearDown(self):
        OpenScenarioParser.set_execution_context(None)
        self._patch.stop()

    @staticmethod
    def _build(root, overrides=None):
        context = create_execution_context(SimpleNamespace(xml_tree=ET.ElementTree(root),
                                                           _custom_params=overrides or {}))
        OpenScenarioParser.set_execution_context(context)
        return context

    def test_boolean_attribute_reference(self):
        root = scenario_root()
        root.find(".//Throttle").set("active", "$Override")
        self._build(root)
        controller = root.find(".//ControllerAction")
        self.assertEqual(OpenScenarioParser.get_controller(controller, {})[0], "external_control")

        self._build(root, {"Override": "true"})
        with self.assertRaises(NotImplementedError):
            OpenScenarioParser.get_controller(controller, {})

    def test_condition_delay_reference(self):
        root = scenario_root()
        condition = root.find(".//Condition[@name='StartTime']")
        condition.set("delay", "$Delay")
        self._build(root)
        atomic = OpenScenarioParser.convert_condition_to_atomic(condition, [None])
        delays = [node for node in atomic.iterate() if isinstance(node, TimeOut)]
        self.assertEqual([node._timeout_value for node in delays], [2.5])  # pylint: disable=protected-access

    def test_unknown_reference_is_an_evaluation_error(self):
        self._build(scenario_root())
        with self.assertRaises(EvaluationError) as raised:
            float(ParameterRef("$nope"))
        self.assertEqual(raised.exception.diagnostic.code, Diagnostic.UNKNOWN_REFERENCE)

    def test_expressions_resolve_without_execution_context(self):
        CarlaDataProvider.update_osc_global_params({"Gap": "25"})
        self.assertEqual(float(ParameterRef("${$Gap * 2}")), 50.0)


class TestStructuredDocumentErrors(unittest.TestCase):

    def test_loader_errors_are_validation_errors(self):
        documents = (
            "<OpenSCENARIO><Storyboard/></OpenSCENARIO>",
            "<Scenario><FileHeader revMajor='1' revMinor='0'/></Scenario>",
            "<OpenSCENARIO><FileHeader revMajor='1'/></OpenSCENARIO>",
            "<OpenSCENARIO><FileHeader revMajor='one' revMinor='0'/></OpenSCENARIO>",
        )
        for document in documents:
            with self.subTest(document=document):
                with self.assertRaises(ValidationError) as raised:
                    OpenScenarioLoader().load(ET.fromstring(document))
                self.assertIsInstance(raised.exception, ValueError)
                self.assertEqual([item.code for item in raised.exception.diagnostics],
                                 [Diagnostic.MALFORMED_DOCUMENT])

    def test_xml_parse_error_is_a_validation_error(self):
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory)
        path = os.path.join(directory, "broken.xosc")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("<OpenSCENARIO><FileHeader")
        with self.assertRaises(ValidationError) as raised:
            OpenScenarioLoader().load(path)
        self.assertEqual(raised.exception.diagnostics[0].code, Diagnostic.MALFORMED_DOCUMENT)

    def test_evaluation_error_is_a_structured_error(self):
        error = EvaluationError("Division by zero", Diagnostic.ARITHMETIC_ERROR, "${1 / 0}")
        self.assertIsInstance(error, DiagnosticError)
        self.assertEqual(error.diagnostics, [error.diagnostic])
        with self.assertRaises(ValidationError) as raised:
            with diagnostics_as_validation_error():
                raise error
        self.assertEqual(raised.exception.diagnostics, [error.diagnostic])


class TestSemanticValidatorReferences(unittest.TestCase):

    @staticmethod
    def _validate(root):
        return SemanticValidator().validate(OpenScenarioLoader().load(root), raise_on_error=False)

    def test_undeclared_reference_in_any_attribute_for_every_version(self):
        for minor in ("0", "3"):
            with self.subTest(revMinor=minor):
                root = scenario_root()
                root.find("FileHeader").set("revMinor", minor)
                root.find(".//RelativeRoadPosition").set("ds", "$nope")
                root.find(".//Sun").set("azimuth", "${$alsoMissing + 1}")
                diagnostics = self._validate(root)
                self.assertEqual(sorted(item.source.tag for item in diagnostics),
                                 ["RelativeRoadPosition", "Sun"])
                self.assertEqual(unknown_codes(diagnostics), [Diagnostic.UNKNOWN_REFERENCE] * 2)

    def test_declared_and_name_attributes_are_accepted(self):
        root = scenario_root()
        root.find(".//RelativeRoadPosition").set("ds", "${$Gap * 2}")
        root.find(".//Sun").set("azimuth", "$leadingSpeed")
        self.assertEqual(self._validate(root), [])

    def test_invalid_reference_syntax_is_reported(self):
        root = scenario_root()
        root.find(".//Sun").set("azimuth", "$1abc")
        self.assertEqual([item.code for item in self._validate(root)], [Diagnostic.INVALID_EXPRESSION])


if __name__ == "__main__":
    unittest.main()
