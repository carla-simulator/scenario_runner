#!/usr/bin/env python

"""Unit tests for the backend-independent OpenSCENARIO foundation."""

from __future__ import absolute_import

import unittest
import xml.etree.ElementTree as ET

from srunner.openscenario.execution.capabilities import BackendCapabilities
from srunner.openscenario.execution.context import ExecutionContext
from srunner.openscenario.execution.registry import ActionExecutorRegistry, UnsupportedFeatureError
from srunner.openscenario.model.actions import ParameterAction
from srunner.openscenario.parsing.loader import OpenScenarioLoader
from srunner.openscenario.validation.semantic import SemanticValidator


DOCUMENT = """<OpenSCENARIO>
  <FileHeader revMajor="1" revMinor="3" revPatch="1" />
  <ParameterDeclarations>
    <ParameterDeclaration name="Speed" parameterType="double" value="10.0" />
  </ParameterDeclarations>
  <Storyboard>
    <Init><Actions><GlobalAction><ParameterAction parameterRef="Speed">
      <SetAction value="12.0" />
    </ParameterAction></GlobalAction></Actions></Init>
  </Storyboard>
</OpenSCENARIO>"""


class TestOpenScenarioFoundation(unittest.TestCase):
    def test_loader_retains_version_and_typed_parameters(self):
        document = OpenScenarioLoader().load(ET.fromstring(DOCUMENT))
        self.assertEqual(str(document.version), "1.3.1")
        self.assertEqual(document.parameter_declarations[0].name, "Speed")
        self.assertEqual(document.parameter_values()["Speed"], "10.0")

    def test_loader_rejects_missing_version(self):
        root = ET.fromstring("<OpenSCENARIO><FileHeader /></OpenSCENARIO>")
        with self.assertRaises(ValueError):
            OpenScenarioLoader().load(root)

    def test_parameter_action_is_semantic_and_dispatchable(self):
        document = OpenScenarioLoader().load(ET.fromstring(DOCUMENT))
        action = document.parameter_actions()[0]
        registry = ActionExecutorRegistry()
        registry.register("ParameterAction", lambda item, context: item.parameter_ref)
        self.assertEqual(registry.execute(action, ExecutionContext()).strip("$"), "Speed")

    def test_unsupported_action_has_structured_diagnostic(self):
        registry = ActionExecutorRegistry()
        with self.assertRaises(UnsupportedFeatureError) as raised:
            registry.execute(ParameterAction("Speed"), ExecutionContext())
        self.assertEqual(raised.exception.diagnostic.category, "unsupported_feature")

    def test_semantic_validation_fails_before_execution(self):
        root = ET.fromstring(DOCUMENT)
        declarations = root.find("ParameterDeclarations")
        declarations.append(ET.fromstring(
            '<ParameterDeclaration name="Speed" parameterType="double" value="11.0" />'))
        document = OpenScenarioLoader().load(root)
        diagnostics = SemanticValidator().validate(document, raise_on_error=False)
        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(diagnostics[0].category, "semantic_error")

    def test_capability_boundary_is_backend_independent(self):
        capabilities = BackendCapabilities(["environment_lights"])
        self.assertTrue(capabilities.supports("environment_lights"))
        self.assertFalse(capabilities.supports("generic_animation"))


if __name__ == "__main__":
    unittest.main()
