#!/usr/bin/env python
"""Integration tests for the semantic OpenSCENARIO runtime wiring.

The behaviours produced by ``OpenScenario`` and ``OpenScenarioParser`` are
built from real ``.xosc`` documents and ticked directly; no CARLA server is
needed, but the ScenarioRunner runtime modules (``carla`` wheel, ``agents``)
must be importable.
"""

from __future__ import absolute_import

import glob
import os
import unittest
import xml.etree.ElementTree as ET
from types import SimpleNamespace
from unittest import mock

try:
    import py_trees
    from srunner.openscenario.compatibility.atomics import ExecuteSemanticAction, SemanticValueCondition
    from srunner.scenarioconfigs.openscenario_configuration import OpenScenarioConfiguration
    from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
    from srunner.scenariomanager.scenarioatomics.atomic_behaviors import ChangeParameter, RunScript
    from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import CheckParameter
    from srunner.scenarios.open_scenario import create_execution_context, create_init_parameters_behavior
    from srunner.tools.openscenario_parser import OpenScenarioParser, ParameterRef
    RUNTIME_IMPORT_ERROR = None
except ImportError as error:  # pragma: no cover - depends on the local CARLA installation
    RUNTIME_IMPORT_ERROR = error

from srunner.openscenario.validation.diagnostics import Diagnostic, ValidationError

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(TESTS_DIR, "fixtures", "openscenario", "parameters_variables.xosc")
EXAMPLES = sorted(glob.glob(os.path.join(TESTS_DIR, "..", "examples", "*.xosc")))
SUCCESS = "SUCCESS"


@unittest.skipIf(RUNTIME_IMPORT_ERROR is not None, "ScenarioRunner runtime not importable: {}".format(
    RUNTIME_IMPORT_ERROR))
class TestOpenScenarioRuntimeWiring(unittest.TestCase):

    def setUp(self):
        self._patch = mock.patch.object(CarlaDataProvider, "_global_osc_parameters", {})
        self.global_parameters = self._patch.start()
        py_trees.blackboard.Blackboard().__dict__.clear()

    def tearDown(self):
        OpenScenarioParser.set_execution_context(None)
        self._patch.stop()

    @staticmethod
    def _root():
        return ET.parse(FIXTURE).getroot()

    def _build(self, root=None, overrides=None):
        config = SimpleNamespace(xml_tree=ET.ElementTree(root if root is not None else self._root()),
                                 _custom_params=overrides or {})
        context = create_execution_context(config)
        OpenScenarioParser.set_execution_context(context)
        return context

    @staticmethod
    def _action(context, name):
        element = context.document.root.find(".//Action[@name='{}']".format(name))
        return OpenScenarioParser.convert_maneuver_to_atomic(element, None, [None], {}, None)

    @staticmethod
    def _condition(context, name):
        element = context.document.root.find(".//Condition[@name='{}']".format(name))
        return OpenScenarioParser.convert_condition_to_atomic(element, [None])

    @staticmethod
    def _tick(behaviour):
        behaviour.tick_once()
        return behaviour.status.name

    def test_variable_action_gates_variable_condition_end_to_end(self):
        context = self._build()
        variables = context.evaluation.variables

        when_ready = self._condition(context, "WhenReady")
        self.assertIsInstance(when_ready, SemanticValueCondition)
        self.assertEqual(self._tick(when_ready), "RUNNING")

        self.assertEqual(self._tick(create_init_parameters_behavior(context)), SUCCESS)
        self.assertEqual(self.global_parameters["Counter"], 10)
        self.assertEqual(variables.get("Count"), 10)
        self.assertIs(variables.get("Ready"), True)

        self.assertEqual(self._tick(when_ready), SUCCESS)
        set_label = self._action(context, "SetLabelAction")
        self.assertIsInstance(set_label, ExecuteSemanticAction)
        self.assertEqual(self._tick(set_label), SUCCESS)
        self.assertEqual(variables.get("Label"), "running")

        self.assertEqual(self._tick(self._condition(context, "WhenCounterSet")), SUCCESS)
        self.assertEqual(self._tick(self._condition(context, "WhenCountReached")), SUCCESS)
        self.assertEqual(self._tick(self._action(context, "BumpCounterAction")), SUCCESS)
        self.assertEqual(self.global_parameters["Counter"], 13)
        self.assertIs(type(self.global_parameters["Counter"]), int)

    def test_legacy_parameter_consumers_see_semantic_state(self):
        context = self._build(overrides={"BaseSpeed": "20"})
        self.assertEqual(self.global_parameters["TargetSpeed"], 41.0)
        self.assertEqual(float(ParameterRef("$TargetSpeed")), 41.0)
        self.assertEqual(float(ParameterRef("${$TargetSpeed + 1}")), 42.0)
        self.assertEqual(CarlaDataProvider.get_osc_global_param_value("$Laps"), 5)
        self._tick(create_init_parameters_behavior(context))
        self.assertEqual(int(ParameterRef("$Counter")), 10)

    def test_unmigrated_action_stays_on_legacy_path(self):
        context = self._build()
        self.assertIsInstance(self._action(context, "RunCommandAction"), RunScript)

        OpenScenarioParser.set_execution_context(None)
        self.assertIsInstance(self._action(context, "BumpCounterAction"), ChangeParameter)
        self.assertIsInstance(self._condition(context, "WhenCounterSet"), CheckParameter)
        self.assertEqual(ParameterRef("${1 + 1}").get_interpreted_value(), 2)
        with self.assertRaises(NotImplementedError):
            self._action(context, "SetLabelAction")
        with self.assertRaises(NotImplementedError):
            self._condition(context, "WhenReady")

    def test_unknown_variable_reference_raises_validation_error(self):
        root = self._root()
        root.find(".//VariableCondition").set("variableRef", "Missing")
        with self.assertRaises(ValidationError) as raised:
            self._build(root)
        diagnostics = raised.exception.diagnostics
        self.assertEqual([item.code for item in diagnostics], [Diagnostic.UNKNOWN_REFERENCE])
        self.assertIn("Missing", diagnostics[0].message)
        self.assertEqual(diagnostics[0].source.tag, "VariableCondition")

    def test_duplicate_declarations_raise_validation_error(self):
        root = self._root()
        root.find("ParameterDeclarations").append(ET.fromstring(
            '<ParameterDeclaration name="Factor" parameterType="int" value="3"/>'))
        with self.assertRaises(ValidationError) as raised:
            self._build(root)
        self.assertEqual([item.code for item in raised.exception.diagnostics], [Diagnostic.DUPLICATE_DECLARATION])

    def test_runtime_evaluation_failure_is_a_behaviour_failure(self):
        root = self._root()
        root.find(".//Init//VariableAction[@variableRef='Ready']/SetAction").set("value", "maybe")
        context = self._build(root)
        init = create_init_parameters_behavior(context)
        self.assertEqual(self._tick(init), "FAILURE")
        failed = [node for node in init.iterate() if isinstance(node, ExecuteSemanticAction) and node.diagnostic]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0].diagnostic.code, Diagnostic.TYPE_MISMATCH)

    def test_configuration_evaluates_expression_declarations(self):
        self.addCleanup(setattr, OpenScenarioParser, "global_osc_parameters",
                        OpenScenarioParser.global_osc_parameters)
        configuration = OpenScenarioConfiguration.__new__(OpenScenarioConfiguration)
        configuration.xml_tree = ET.parse(FIXTURE)
        configuration._custom_params = {"Factor": "3"}  # pylint: disable=protected-access
        configuration._set_parameters()  # pylint: disable=protected-access
        self.assertEqual(float(ParameterRef("$TargetSpeed")), 31.0)
        self.assertEqual(OpenScenarioParser.global_osc_parameters["TargetSpeed"], 31.0)

    def test_shipped_examples_pass_semantic_validation(self):
        self.assertTrue(EXAMPLES)
        for path in EXAMPLES:
            with self.subTest(example=os.path.basename(path)):
                config = SimpleNamespace(xml_tree=ET.parse(path), _custom_params={})
                context = create_execution_context(config)
                self.assertEqual(context.errors, [])


if __name__ == "__main__":
    unittest.main()
