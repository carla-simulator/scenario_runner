"""Backend-independent OpenSCENARIO document model."""

from __future__ import absolute_import

import copy

from srunner.openscenario.model.actions import (
    AcquirePositionAction, AssignRouteAction, FollowTrajectoryAction, ParameterAction, SpeedProfileAction,
    VariableModifyAction, VariableSetAction)
from srunner.openscenario.model.parameters import (
    ParameterDeclaration, ParameterStore, VariableDeclaration, VariableStore)
from srunner.openscenario.model.conditions import ENTITY_CONDITIONS
from srunner.openscenario.validation.diagnostics import ValidationError
from srunner.openscenario.model.references import (
    FOLLOWING_MODES, CatalogReference, Position, Route, TimeReference, Trajectory, following_mode_from_xml,
    invalid)


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
        """Build a semantic ParameterAction from a ``ParameterAction`` element.

        Raises ValidationError when the element has neither a SetAction nor a ModifyAction with a
        valid Rule.
        """
        set_action = element.find("SetAction")
        modify_action = element.find("ModifyAction")
        parameter_ref = element.attrib.get("parameterRef")
        if set_action is not None:
            return ParameterAction(parameter_ref=parameter_ref, set_value=set_action.attrib.get("value"),
                                   source=element)
        if modify_action is None:
            raise invalid("ParameterAction {!r} requires a SetAction or ModifyAction".format(parameter_ref), element)
        rule = modify_action.find("Rule")
        for tag, operator_symbol in (("AddValue", "+"), ("MultiplyByValue", "*")):
            value = rule.find(tag) if rule is not None else None
            if value is not None:
                return ParameterAction(parameter_ref=parameter_ref, modify_rule=operator_symbol,
                                       modify_value=value.attrib.get("value"), source=element)
        raise invalid("ModifyAction of ParameterAction {!r} requires a Rule with AddValue or MultiplyByValue".format(
            parameter_ref), modify_action)

    @staticmethod
    def parse_variable_action(element):
        """Build a semantic variable action from a ``VariableAction`` element; invalid ones raise ValidationError."""
        variable_ref = element.attrib.get("variableRef")
        set_action = element.find("SetAction")
        modify_action = element.find("ModifyAction")
        if set_action is not None:
            return VariableSetAction(variable_ref, set_action.attrib.get("value"), element)
        if modify_action is None:
            raise invalid("VariableAction {!r} requires a SetAction or ModifyAction".format(variable_ref), element)
        rule = modify_action.find("Rule")
        for tag, rule_name in (("AddValue", "add"), ("MultiplyByValue", "multiply")):
            value = rule.find(tag) if rule is not None else None
            if value is not None:
                return VariableModifyAction(variable_ref, rule_name, value.attrib.get("value"), element)
        raise invalid("ModifyAction of VariableAction {!r} requires a Rule with AddValue or MultiplyByValue".format(
            variable_ref), modify_action)

    def _actions(self, tag, parser, include_story, diagnostics=None):
        actions = []
        for index, global_action in enumerate(self._global_actions(include_story)):
            element = global_action.find(tag)
            if element is None:
                continue
            try:
                action = parser(element)
            except ValidationError as error:
                if diagnostics is None:
                    raise
                diagnostics.extend(error.diagnostics)
                continue
            if action is not None:
                action.sequence_index = index
                actions.append(action)
        return actions

    def parameter_actions(self, include_story=False, diagnostics=None):
        """Return semantic ParameterActions from Init (and the stories, if requested).

        Invalid actions raise ValidationError, or are reported into ``diagnostics`` when given.
        """
        return self._actions("ParameterAction", self.parse_parameter_action, include_story, diagnostics)

    def variable_actions(self, include_story=False, diagnostics=None):
        """Return semantic VariableActions from Init (and the stories, if requested)."""
        return self._actions("VariableAction", self.parse_variable_action, include_story, diagnostics)

    def init_global_actions(self):
        """Return Init Parameter/Variable actions in document order."""
        actions = self.parameter_actions() + self.variable_actions()
        return sorted(actions, key=lambda action: action.sequence_index)

    def value_conditions(self, tag):
        """Return ``ParameterCondition``/``VariableCondition`` elements found in the document."""
        return list(self.root.iter(tag))

    def positions(self):
        return [Position.from_xml(element) for element in self.root.iter("Position")]

    def routes(self):
        return [Route.from_xml(element) for element in self.root.iter("Route")]

    def trajectories(self):
        return [Trajectory.from_xml(element) for element in self.root.iter("Trajectory")]

    @staticmethod
    def parse_assign_route_action(element):
        route = element.find("Route")
        reference = element.find("CatalogReference")
        if (route is None) == (reference is None):
            raise invalid("AssignRouteAction requires exactly one of Route or CatalogReference", element)
        return AssignRouteAction(route=Route.from_xml(route) if route is not None else None,
                                 catalog_reference=CatalogReference.from_xml(reference), source=element)

    @staticmethod
    def parse_acquire_position_action(element):
        return AcquirePositionAction(Position.from_xml(element.find("Position")), source=element)

    @staticmethod
    def parse_follow_trajectory_action(element):
        container = element.find("TrajectoryRef")
        legacy = [child for child in (element.find("Trajectory"), element.find("CatalogReference"))
                  if child is not None]
        if container is not None:
            if legacy:
                raise invalid("FollowTrajectoryAction must not combine TrajectoryRef with a deprecated "
                              "Trajectory or CatalogReference", element)
            targets = list(container)
        else:
            targets = legacy
        if len(targets) != 1 or targets[0].tag not in ("Trajectory", "CatalogReference"):
            raise invalid("FollowTrajectoryAction requires exactly one Trajectory or CatalogReference", element)
        target = targets[0]
        return FollowTrajectoryAction(
            trajectory=Trajectory.from_xml(target) if target.tag == "Trajectory" else None,
            catalog_reference=CatalogReference.from_xml(target) if target.tag == "CatalogReference" else None,
            time_reference=TimeReference.from_xml(element.find("TimeReference")),
            following_mode=following_mode_from_xml(element.find("TrajectoryFollowingMode")),
            initial_distance_offset=element.attrib.get("initialDistanceOffset"),
            source=element)

    @staticmethod
    def parse_speed_profile_action(element):
        entries = []
        for entry in element.findall("SpeedProfileEntry"):
            if "speed" not in entry.attrib:
                raise invalid("SpeedProfileEntry requires a speed", entry)
            entries.append((entry.attrib["speed"], entry.attrib.get("time")))
        if not entries:
            raise invalid("SpeedProfileAction requires at least one SpeedProfileEntry", element)
        mode = element.attrib.get("followingMode")
        if mode not in FOLLOWING_MODES and not (mode or "").startswith("$"):
            raise invalid("SpeedProfileAction followingMode must be 'follow' or 'position', got {!r}".format(
                mode), element)
        constraints = element.find("DynamicConstraints")
        return SpeedProfileAction(entries, mode, element.attrib.get("entityRef"),
                                  dict(constraints.attrib) if constraints is not None else None, element)

    _PRIVATE_ACTION_PARSERS = {
        "AssignRouteAction": "parse_assign_route_action",
        "AcquirePositionAction": "parse_acquire_position_action",
        "FollowTrajectoryAction": "parse_follow_trajectory_action",
        "SpeedProfileAction": "parse_speed_profile_action",
    }

    @classmethod
    def parse_private_action(cls, element):
        """Build a typed action from a migrated private action element, or None if it is not migrated.

        ``element`` may be the action itself, its RoutingAction/LongitudinalAction container, or the
        enclosing PrivateAction. Structurally invalid elements raise ValidationError.
        """
        if element is None:
            return None
        if element.tag in ("PrivateAction", "RoutingAction", "LongitudinalAction"):
            for child in element:
                action = cls.parse_private_action(child)
                if action is not None or child.tag in ("RoutingAction", "LongitudinalAction"):
                    return action
            return None
        parser = cls._PRIVATE_ACTION_PARSERS.get(element.tag)
        return getattr(cls, parser)(element) if parser is not None else None

    def routing_actions(self):
        """Return typed routing actions of the whole document in document order."""
        return [self.parse_private_action(element) for element in self.root.iter()
                if element.tag in ("AssignRouteAction", "AcquirePositionAction", "FollowTrajectoryAction")]

    def speed_profile_actions(self):
        return [self.parse_speed_profile_action(element) for element in self.root.iter("SpeedProfileAction")]

    @staticmethod
    def parse_entity_condition(element, name=None, edge=None):
        """Build a typed condition from a migrated EntityCondition child, or None if it is not migrated."""
        if element is None:
            return None
        if element.tag == "EntityCondition":
            children = list(element)
            return OpenScenarioDocument.parse_entity_condition(children[0], name, edge) if children else None
        condition_type = ENTITY_CONDITIONS.get(element.tag)
        return condition_type.from_xml(element, name, edge) if condition_type is not None else None

    def entity_conditions(self):
        conditions = []
        for condition in self.root.iter("Condition"):
            for entity_condition in condition.iter("EntityCondition"):
                parsed = self.parse_entity_condition(entity_condition, condition.attrib.get("name"),
                                                     condition.attrib.get("conditionEdge"))
                if parsed is not None:
                    conditions.append(parsed)
        return conditions

    def copy_xml(self):
        return copy.deepcopy(self.root)
