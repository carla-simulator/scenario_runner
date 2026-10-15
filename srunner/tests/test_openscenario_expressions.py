#!/usr/bin/env python
"""CARLA-free tests for OpenSCENARIO expression evaluation and typed parameter resolution."""

from __future__ import absolute_import

import os
import unittest
import xml.etree.ElementTree as ET

from srunner.openscenario.execution.context import ExecutionContext
from srunner.openscenario.execution.registry import default_registry
from srunner.openscenario.model.expressions import EvaluationError, UnknownReferenceError, check_expression, evaluate
from srunner.openscenario.model.parameters import (
    EvaluationContext, ParameterDeclaration, ParameterStore, VariableDeclaration, VariableStore)
from srunner.openscenario.parsing.loader import load_document
from srunner.openscenario.validation.diagnostics import Diagnostic

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "openscenario", "parameters_variables.xosc")


def _context():
    parameters = ParameterStore([
        ParameterDeclaration("a", "int", "3"),
        ParameterDeclaration("b", "double", "2.5"),
        ParameterDeclaration("flag", "boolean", "true"),
        ParameterDeclaration("name", "string", "ego"),
    ])
    variables = VariableStore([VariableDeclaration("v", "int", "7")])
    return EvaluationContext(parameters, variables)


class TestExpressionEvaluator(unittest.TestCase):

    def test_arithmetic_and_functions(self):
        context = _context()
        cases = {
            "${$a + 2 * $b}": 8.0,
            "${($a + 1) * 2}": 8,
            "${pow(2, 8) - 1}": 255.0,
            "${-round(2.6)}": -3,
            "${round(2.5)}": 3,
            "${floor(-1.5)}": -2,
            "${ceil(1.2)}": 2,
            "${1 + sqrt(9) * 2.2}": 1 + 3.0 * 2.2,
            "${abs(-$a)}": 3,
            "${sign(-$b)}": -1.0,
            "${max($a, 4)}": 4,
            "${min($a, $b)}": 2.5,
            "${7 % 3}": 1,
            "${-7 % 3}": -1,
            "${$a / 2}": 1.5,
            "${not $flag or $flag and false}": False,
            "${not 0}": True,
            "${atan(0)}": 0.0,
        }
        for expression, expected in cases.items():
            with self.subTest(expression=expression):
                result = context.resolve(expression)
                self.assertEqual(result, expected)
                self.assertIs(type(result), type(expected))

    def test_declared_types_are_preserved(self):
        context = _context()
        self.assertIs(type(context.resolve("${$a * 2}", "int")), int)
        self.assertEqual(context.resolve("${$a * 2}", "double"), 6.0)
        self.assertIs(type(context.resolve("${$a * 2}", "double")), float)
        self.assertEqual(context.resolve("$a"), 3)
        self.assertEqual(context.resolve("$b", "double"), 2.5)
        self.assertEqual(context.resolve("${round($b)}", "unsignedInt"), 3)
        self.assertIs(context.resolve("${not $flag}", "boolean"), False)
        self.assertEqual(context.resolve("${$a + 1}", "string"), "4")
        self.assertEqual(context.resolve("$name"), "ego")

    def test_type_mismatches_are_errors(self):
        context = _context()
        for expression, expected_type in (("${$b * 2}", "int"), ("${$a / 3}", "int"), ("${$a + 1}", "boolean"),
                                          ("${-1}", "unsignedInt"), ("${$flag + 1}", None),
                                          ("${$name + 1}", None), ("${not $a}", None)):
            with self.subTest(expression=expression):
                with self.assertRaises(EvaluationError) as raised:
                    context.resolve(expression, expected_type)
                self.assertEqual(raised.exception.diagnostic.code, Diagnostic.TYPE_MISMATCH)

    def test_arithmetic_errors(self):
        context = _context()
        for expression in ("${1 / 0}", "${$a % 0}", "${sqrt(-1)}", "${pow(10, 400)}", "${asin(2)}"):
            with self.subTest(expression=expression):
                with self.assertRaises(EvaluationError) as raised:
                    context.resolve(expression)
                self.assertEqual(raised.exception.diagnostic.code, Diagnostic.ARITHMETIC_ERROR)

    def test_malicious_and_unsupported_input_is_rejected(self):
        context = _context()
        for expression in ('${__import__("os").system("true")}', "${().__class__}", "${$a ** 2}",
                           "${$a // 2}", "${$a < 2}", "${lambda: 1}", "${'text'}", "${open('x')}",
                           "${round(1, 2)}", "${round(x=1)}", "${a + 1}", "${_osc_ref_a}", "${}",
                           "${1 if 1 else 2}", "${[1][0]}", "${" + "-" * 2000 + "1}", "${(1)" + ")" * 3 + "}",
                           "${True}", "${1j}"):
            with self.subTest(expression=expression):
                with self.assertRaises(EvaluationError) as raised:
                    context.resolve(expression)
                self.assertIsInstance(raised.exception.diagnostic, Diagnostic)

    def test_missing_references_are_structured(self):
        context = _context()
        for expression in ("$missing", "${$missing + 1}"):
            with self.subTest(expression=expression):
                with self.assertRaises(UnknownReferenceError) as raised:
                    context.resolve(expression)
                self.assertIsInstance(raised.exception, KeyError)
                self.assertEqual(raised.exception.diagnostic.code, Diagnostic.UNKNOWN_REFERENCE)
                self.assertEqual(raised.exception.name, "missing")

    def test_variables_are_not_visible_to_expressions(self):
        context = _context()
        with self.assertRaises(UnknownReferenceError):
            context.resolve("${$v + 1}")
        self.assertTrue(context.compare_variable("v", "equalTo", "${$a + 4}"))

    def test_static_check_reports_references_without_lookup(self):
        self.assertEqual(check_expression("${$x * 2 + max($y, 1)}"), ["x", "y"])
        with self.assertRaises(EvaluationError):
            check_expression("${$x ** 2}")
        self.assertEqual(evaluate("${2 * 3}", lambda name: None), 6)


class TestParameterStores(unittest.TestCase):

    def test_duplicate_declarations_are_recorded(self):
        store = ParameterStore([ParameterDeclaration("Speed", "double", "1"),
                                ParameterDeclaration("Speed", "double", "2")])
        self.assertEqual(len(store.diagnostics), 1)
        self.assertEqual(store.diagnostics[0].code, Diagnostic.DUPLICATE_DECLARATION)
        self.assertEqual(store.get("Speed"), 2.0)

    def test_binding_writes_through_and_reads_through(self):
        external = {"Unrelated": "x"}
        store = ParameterStore([ParameterDeclaration("Speed", "double", "1.5")], binding=external)
        self.assertEqual(external["Speed"], "1.5")
        store.set("Speed", "3")
        self.assertEqual(external["Speed"], 3.0)
        external["Speed"] = "4.5"
        self.assertEqual(store.get("Speed"), 4.5)
        self.assertEqual(store.resolve("$Unrelated"), "x")

    def test_variable_modification_rules(self):
        variables = VariableStore([VariableDeclaration("count", "int", "2"),
                                   VariableDeclaration("label", "string", "a")])
        self.assertEqual(variables.modify("count", "add", "3"), 5)
        self.assertEqual(variables.modify("count", "multiply", 2), 10)
        with self.assertRaises(EvaluationError):
            variables.modify("count", "add", "1.5")
        with self.assertRaises(EvaluationError):
            variables.modify("label", "add", "b")
        with self.assertRaises(EvaluationError):
            variables.modify("count", "power", "2")


class TestExecutionContextForDocument(unittest.TestCase):

    def test_document_stores_are_linked_and_initialised(self):
        context = ExecutionContext.for_document(load_document(FIXTURE))
        self.assertEqual(context.errors, [])
        self.assertIs(context.evaluation.parameters, context.document.parameter_store)
        self.assertIs(context.evaluation.variables, context.document.variable_store)
        self.assertEqual(context.evaluation.resolve("$TargetSpeed"), 21.0)
        self.assertEqual(context.evaluation.resolve("$Laps"), 3)
        self.assertEqual(context.evaluation.variables.get("Count"), 3)

    def test_overrides_and_binding(self):
        runtime_parameters = {}
        context = ExecutionContext.for_document(
            load_document(FIXTURE), parameter_overrides={"BaseSpeed": "20", "Unknown": "1"},
            parameter_binding=runtime_parameters)
        self.assertEqual(context.errors, [])
        self.assertEqual([item.code for item in context.diagnostics], [Diagnostic.UNDECLARED_OVERRIDE])
        self.assertEqual(runtime_parameters["BaseSpeed"], 20.0)
        self.assertEqual(runtime_parameters["TargetSpeed"], 41.0)

    def test_invalid_override_is_a_diagnostic(self):
        context = ExecutionContext.for_document(load_document(FIXTURE), parameter_overrides={"Factor": "two"})
        self.assertEqual([item.code for item in context.errors], [Diagnostic.TYPE_MISMATCH])

    def test_init_actions_execute_in_document_order(self):
        context = ExecutionContext.for_document(load_document(FIXTURE))
        registry = default_registry()
        for action in context.document.init_global_actions():
            registry.execute(action, context)
        self.assertEqual(context.evaluation.parameters.get("Counter"), 10)
        self.assertEqual(context.evaluation.variables.get("Count"), 10)
        self.assertIs(context.evaluation.variables.get("Ready"), True)

    def test_semantic_errors_are_collected(self):
        root = ET.parse(FIXTURE).getroot()
        root.find("VariableDeclarations").append(ET.fromstring(
            '<VariableDeclaration name="Ready" variableType="boolean" value="true"/>'))
        condition = root.find(".//VariableCondition")
        condition.set("variableRef", "NotDeclared")
        condition.set("value", "${$Factor ** 2}")
        root.find(".//ParameterCondition").set("value", "$Count")
        context = ExecutionContext.for_document(load_document(root))
        codes = sorted(item.code for item in context.errors)
        self.assertEqual(codes, sorted([Diagnostic.DUPLICATE_DECLARATION, Diagnostic.UNKNOWN_REFERENCE,
                                        Diagnostic.INVALID_EXPRESSION, Diagnostic.UNKNOWN_REFERENCE]))
        self.assertIn("variables can only be read by a VariableCondition",
                      " ".join(item.message for item in context.errors))


if __name__ == "__main__":
    unittest.main()
