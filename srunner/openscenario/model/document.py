"""Backend-independent OpenSCENARIO document model."""

from __future__ import absolute_import

import copy

from srunner.openscenario.model.actions import ParameterAction, VariableModifyAction, VariableSetAction
from srunner.openscenario.model.parameters import (
    ParameterDeclaration, ParameterStore, VariableDeclaration, VariableStore)
from srunner.openscenario.model.references import CatalogReference, Position


class OpenScenarioDocument(object):
    """A parsed document retaining XML and typed semantic declarations."""

    def __init__(self, root, version, source=None):
        self.root = root
        self.version = version
        self.source = source
        self.parameter_declarations = self._read_parameters(root)
        self.variable_declarations = self._read_variables(root)
        self.catalog_references = self._read_catalog_references(root)
        self.parameter_store = self._make_store(self.parameter_declarations)
        self.variable_store = self._make_variable_store(self.variable_declarations)

    @staticmethod
    def _read_parameters(root):
        declarations = []
        container = root.find("ParameterDeclarations")
        if container is None:
            return declarations
        for element in container.findall("ParameterDeclaration"):
            declarations.append(ParameterDeclaration(
                name=element.attrib.get("name"),
                parameter_type=element.attrib.get("parameterType"),
                value=element.attrib.get("value"),
                source=element,
            ))
        return declarations

    def parameter_values(self, overrides=None):
        values = dict((item.name, item.value) for item in self.parameter_declarations)
        if overrides:
            values.update(overrides)
        return values

    @staticmethod
    def _read_variables(root):
        declarations = []
        containers = [root.find("VariableDeclarations")]
        storyboard = root.find("Storyboard")
        if storyboard is not None:
            containers.append(storyboard.find("VariableDeclarations"))
        for container in containers:
            if container is None:
                continue
            for element in container.findall("VariableDeclaration"):
                declarations.append(VariableDeclaration(
                    name=element.attrib.get("name"),
                    variable_type=element.attrib.get("variableType", element.attrib.get("type")),
                    value=element.attrib.get("value"), source=element))
        return declarations

    @staticmethod
    def _make_store(declarations):
        return ParameterStore(declarations)

    @staticmethod
    def _make_variable_store(declarations):
        return VariableStore(declarations)

    @staticmethod
    def _read_catalog_references(root):
        references = []
        for element in root.iter("CatalogReference"):
            assignments = {}
            container = element.find("ParameterAssignments")
            if container is not None:
                for assignment in container.findall("ParameterAssignment"):
                    assignments[assignment.attrib.get("parameterRef")] = assignment.attrib.get("value")
            references.append(CatalogReference(
                catalog_name=element.attrib.get("catalogName"),
                entry_name=element.attrib.get("entryName"),
                parameter_assignments=assignments,
                source=element,
            ))
        return references

    def _global_actions(self, include_story=False):
        storyboard = self.root.find("Storyboard")
        if storyboard is None:
            return []
        if include_story:
            return list(storyboard.iter("GlobalAction"))
        init = storyboard.find("Init")
        if init is None or init.find("Actions") is None:
            return []
        return list(init.find("Actions").iter("GlobalAction"))

    @staticmethod
    def parse_parameter_action(element):
        """Build a semantic ParameterAction from a ``ParameterAction`` element, or None."""
        set_action = element.find("SetAction")
        modify_action = element.find("ModifyAction")
        if set_action is not None:
            return ParameterAction(parameter_ref=element.attrib.get("parameterRef"),
                                   set_value=set_action.attrib.get("value"), source=element)
        rule = modify_action.find("Rule") if modify_action is not None else None
        if rule is None:
            return None
        for tag, operator_symbol in (("AddValue", "+"), ("MultiplyByValue", "*")):
            value = rule.find(tag)
            if value is not None:
                return ParameterAction(parameter_ref=element.attrib.get("parameterRef"),
                                       modify_rule=operator_symbol, modify_value=value.attrib.get("value"),
                                       source=element)
        return None

    @staticmethod
    def parse_variable_action(element):
        """Build a semantic variable action from a ``VariableAction`` element, or None."""
        variable_ref = element.attrib.get("variableRef")
        set_action = element.find("SetAction")
        modify_action = element.find("ModifyAction")
        if set_action is not None:
            return VariableSetAction(variable_ref, set_action.attrib.get("value"), element)
        rule = modify_action.find("Rule") if modify_action is not None else None
        if rule is None:
            return None
        for tag, rule_name in (("AddValue", "add"), ("MultiplyByValue", "multiply")):
            value = rule.find(tag)
            if value is not None:
                return VariableModifyAction(variable_ref, rule_name, value.attrib.get("value"), element)
        return None

    def _actions(self, tag, parser, include_story):
        actions = []
        for index, global_action in enumerate(self._global_actions(include_story)):
            element = global_action.find(tag)
            if element is None:
                continue
            action = parser(element)
            if action is not None:
                action.sequence_index = index
                actions.append(action)
        return actions

    def parameter_actions(self, include_story=False):
        """Return semantic ParameterActions from Init (and the stories, if requested)."""
        return self._actions("ParameterAction", self.parse_parameter_action, include_story)

    def variable_actions(self, include_story=False):
        """Return semantic VariableActions from Init (and the stories, if requested)."""
        return self._actions("VariableAction", self.parse_variable_action, include_story)

    def init_global_actions(self):
        """Return Init Parameter/Variable actions in document order."""
        actions = self.parameter_actions() + self.variable_actions()
        return sorted(actions, key=lambda action: action.sequence_index)

    def value_conditions(self, tag):
        """Return ``ParameterCondition``/``VariableCondition`` elements found in the document."""
        return list(self.root.iter(tag))

    def positions(self):
        return [Position.from_xml(element) for element in self.root.iter("Position")]

    def copy_xml(self):
        return copy.deepcopy(self.root)
