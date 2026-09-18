"""Backend-independent OpenSCENARIO document model."""

from __future__ import absolute_import

import copy

from srunner.openscenario.model.actions import (
    AcquirePositionAction, AssignRouteAction, FollowTrajectoryAction, ParameterAction, SpeedProfileAction,
    VariableModifyAction, VariableSetAction)
from srunner.openscenario.model.parameters import (
    ParameterDeclaration, ParameterStore, VariableDeclaration, VariableStore)
from srunner.openscenario.model.references import (
    CatalogReference, Position, RelativeAngleCondition, RelativeClearanceCondition, Route, Trajectory)


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

    def routes(self):
        return [Route.from_xml(element) for element in self.root.iter("Route")]

    def trajectories(self):
        return [Trajectory.from_xml(element) for element in self.root.iter("Trajectory")]

    @staticmethod
    def _catalog_reference(element):
        if element is None:
            return None
        return CatalogReference(element.attrib.get("catalogName"), element.attrib.get("entryName"),
                                source=element)

    @staticmethod
    def _time_reference(element):
        if element is None:
            return None
        timing = element.find("Timing")
        if timing is not None:
            return {"kind": "Timing", "attributes": dict(timing.attrib), "source": element}
        return {"kind": "None", "attributes": {}, "source": element}

    @staticmethod
    def _following_mode(element):
        if element is None:
            return None
        return element.attrib.get("followingMode")

    def routing_actions(self):
        """Yield typed routing actions; legacy execution remains available."""
        actions = []
        for element in self.root.iter("AssignRouteAction"):
            route = element.find("Route")
            actions.append(AssignRouteAction(
                route=Route.from_xml(route) if route is not None else None,
                catalog_reference=self._catalog_reference(element.find("CatalogReference")),
                source=element))
        for element in self.root.iter("AcquirePositionAction"):
            position = element.find("Position")
            if position is not None:
                actions.append(AcquirePositionAction(Position.from_xml(position), element))
        for element in self.root.iter("FollowTrajectoryAction"):
            trajectory = element.find("Trajectory")
            actions.append(FollowTrajectoryAction(
                trajectory=Trajectory.from_xml(trajectory,
                                               self._time_reference(element.find("TimeReference")),
                                               self._following_mode(element.find("TrajectoryFollowingMode")))
                if trajectory is not None else None,
                catalog_reference=self._catalog_reference(element.find("CatalogReference")),
                time_reference=self._time_reference(element.find("TimeReference")),
                following_mode=self._following_mode(element.find("TrajectoryFollowingMode")),
                source=element))
        return actions

    @staticmethod
    def action_from_xml(element):
        """Create one semantic action from an action XML element.

        This narrow entry point is used by the incremental runtime adapter so
        migrated actions do not need to be rediscovered through the legacy
        parser's global XML traversal.
        """
        if element is None:
            return None
        if element.tag == "AssignRouteAction":
            route = element.find("Route")
            reference = element.find("CatalogReference")
            return AssignRouteAction(
                route=Route.from_xml(route) if route is not None else None,
                catalog_reference=CatalogReference(
                    reference.attrib.get("catalogName"), reference.attrib.get("entryName"),
                    source=reference) if reference is not None else None,
                source=element)
        if element.tag == "AcquirePositionAction":
            position = element.find("Position")
            return AcquirePositionAction(Position.from_xml(position), source=element)
        if element.tag == "FollowTrajectoryAction":
            trajectory = element.find("Trajectory")
            reference = element.find("CatalogReference")
            return FollowTrajectoryAction(
                trajectory=Trajectory.from_xml(
                    trajectory,
                    time_reference=element.attrib.get("timeReference"),
                    following_mode=element.attrib.get("followingMode")) if trajectory is not None else None,
                catalog_reference=CatalogReference(
                    reference.attrib.get("catalogName"), reference.attrib.get("entryName"),
                    source=reference) if reference is not None else None,
                time_reference=element.attrib.get("timeReference"),
                following_mode=element.attrib.get("followingMode"),
                source=element)
        if element.tag == "SpeedProfileAction":
            segments = [dict(child.attrib) for child in list(element)
                        if child.tag in ("SpeedProfileEntry", "Entry", "SpeedSegment")]
            return SpeedProfileAction(segments, element.attrib.get("entityRef"), element)
        return None

    def speed_profile_actions(self):
        actions = []
        for element in self.root.iter("SpeedProfileAction"):
            segments = []
            for child in list(element):
                if child.tag in ("SpeedProfileEntry", "Entry", "SpeedSegment"):
                    segments.append(dict(child.attrib))
            actions.append(SpeedProfileAction(segments, element.attrib.get("entityRef"), element))
        return actions

    def relative_conditions(self):
        conditions = []
        for element in self.root.iter("RelativeAngleCondition"):
            conditions.append(RelativeAngleCondition(
                element.attrib.get("entityRef"), element.attrib.get("angle"), element.attrib.get("rule"),
                element.attrib.get("coordinateSystem", "entity"), element.attrib.get("name"),
                element.attrib.get("edge"), element))
        for element in self.root.iter("RelativeClearanceCondition"):
            conditions.append(RelativeClearanceCondition(
                element.attrib.get("entityRef"), element.attrib.get("distance"),
                element.attrib.get("freespace", "false").lower() == "true", element.attrib.get("rule"),
                element.attrib.get("distanceType", "cartesianDistance"), element.attrib.get("name"),
                element.attrib.get("edge"), element))
        return conditions

    def copy_xml(self):
        return copy.deepcopy(self.root)
