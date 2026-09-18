"""Backend-independent OpenSCENARIO document model."""

from __future__ import absolute_import

import copy

from srunner.openscenario.model.actions import ParameterAction
from srunner.openscenario.model.parameters import ParameterDeclaration, VariableDeclaration
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
        from srunner.openscenario.model.parameters import ParameterStore
        return ParameterStore(declarations)

    @staticmethod
    def _make_variable_store(declarations):
        from srunner.openscenario.model.parameters import VariableStore
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

    def parameter_actions(self):
        """Yield semantic ParameterActions from the Init storyboard.

        This is intentionally narrow: other actions continue through the legacy
        adapter until they are migrated one feature at a time.
        """
        storyboard = self.root.find("Storyboard")
        if storyboard is None:
            return []
        init = storyboard.find("Init")
        if init is None:
            return []
        actions = []
        actions_container = init.find("Actions")
        if actions_container is None:
            return []
        for index, global_action in enumerate(actions_container.iter("GlobalAction")):
            element = global_action.find("ParameterAction")
            if element is None:
                continue
            set_action = element.find("SetAction")
            modify_action = element.find("ModifyAction")
            if set_action is not None:
                action = ParameterAction(
                    parameter_ref=element.attrib.get("parameterRef"),
                    set_value=set_action.attrib.get("value"),
                    source=element,
                )
                action.sequence_index = index
                actions.append(action)
            elif modify_action is not None:
                rule = modify_action.find("Rule")
                if rule is None:
                    continue
                add = rule.find("AddValue")
                multiply = rule.find("MultiplyByValue")
                if add is not None:
                    action = ParameterAction(
                        parameter_ref=element.attrib.get("parameterRef"),
                        modify_rule="+",
                        modify_value=add.attrib.get("value"),
                        source=element,
                    )
                    action.sequence_index = index
                    actions.append(action)
                elif multiply is not None:
                    action = ParameterAction(
                        parameter_ref=element.attrib.get("parameterRef"),
                        modify_rule="*",
                        modify_value=multiply.attrib.get("value"),
                        source=element,
                    )
                    action.sequence_index = index
                    actions.append(action)
        return actions

    def variable_actions(self):
        """Yield variable actions from Init without resolving them through CARLA."""
        from srunner.openscenario.model.actions import VariableModifyAction, VariableSetAction
        storyboard = self.root.find("Storyboard")
        if storyboard is None or storyboard.find("Init") is None:
            return []
        actions = []
        for index, global_action in enumerate(storyboard.find("Init").iter("GlobalAction")):
            element = global_action.find("VariableAction")
            if element is None:
                continue
            set_action = element.find("VariableSetAction")
            modify_action = element.find("VariableModifyAction")
            if set_action is not None:
                action = VariableSetAction(set_action.attrib.get("variableRef"), set_action.attrib.get("value"), element)
            elif modify_action is not None:
                action = VariableModifyAction(modify_action.attrib.get("variableRef"),
                                              modify_action.attrib.get("modification"),
                                              modify_action.attrib.get("value"), element)
            else:
                continue
            action.sequence_index = index
            actions.append(action)
        return actions

    def positions(self):
        return [Position.from_xml(element) for element in self.root.iter("Position")]

    def copy_xml(self):
        return copy.deepcopy(self.root)
