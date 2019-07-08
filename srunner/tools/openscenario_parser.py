#!/usr/bin/env python

# Copyright (c) 2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides a parser for scenario configuration files
"""

import os
import xml.etree.ElementTree as ET

import xmlschema

import carla
from agents.navigation.local_planner import RoadOption

from srunner.scenariomanager.atomic_scenario_behavior import *
from srunner.scenariomanager.atomic_scenario_criteria import *
from srunner.scenariomanager.timer import TimeOut
from srunner.tools.config_parser import ActorConfigurationData, ScenarioConfiguration


class OpenScenarioConfiguration(ScenarioConfiguration):

    """
    Limitations:
    - Only one Story + Init is supported per Storyboard
    """

    def __init__(self, filename):

        self.xml_tree = ET.parse(filename)

        self._validate_openscenario_configuration()

        self.other_actors = []
        self.ego_vehicles = []
        self.trigger_points = []

        self.storyboard = self.xml_tree.find("Storyboard")
        self.story = self.storyboard.find("Story")
        self.criteria = self.storyboard.find("EndConditions")
        self.init = self.storyboard.find("Init")

        self._parse_openscenario_configuration()

    def _validate_openscenario_configuration(self):
        """
        Validate the given OpenScenario config against the 0.9.1 XSD

        Note: This will throw if the config is not valid. But this is fine here.
        """
        xsd_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../openscenario/OpenSCENARIO_v0.9.1.xsd")
        xsd = xmlschema.XMLSchema(xsd_file)
        xsd.validate(self.xml_tree)

    def _parse_openscenario_configuration(self):
        """
        Parse the given OpenScenario config file, set and validate parameters
        """
        self._set_scenario_name(self.xml_tree)
        self._set_carla_town(self.xml_tree)
        self._set_actor_information(self.xml_tree)

        self._validate_result()

    def _set_scenario_name(self, xml_tree):
        """
        Extract the scenario name from the OpenScenario header information
        """
        header = xml_tree.find("FileHeader")
        self.name = header.attrib.get('description', 'Unknown')

    def _set_carla_town(self, xml_tree):
        """
        Extract the CARLA town (level) from the RoadNetwork information from OpenScenario

        Note: The specification allows multiple Logics elements within the RoadNetwork element.
              Hence, there can be multiple towns specified. We just use the _last_ one.
        """
        for logic in xml_tree.find("RoadNetwork").findall("Logics"):
            self.town = logic.attrib.get('filepath', None)

    def _set_actor_information(self, xml_tree):
        """
        Extract all actors and their corresponding specification
        """
        for entity in xml_tree.iter("Entities"):
            for obj in entity.iter("Object"):
                for vehicle in obj.iter("Vehicle"):
                    model = vehicle.attrib.get('name', "vehicle.*")
                    rolename = 'simulation'
                    ego_vehicle = False
                    for prop in obj.iter("Property"):
                        if prop.get('name', '') == 'rolename':
                            rolename = prop.get('value', 'simulation')
                        if prop.get('name', '') == 'type':
                            ego_vehicle = prop.get('value') == 'ego_vehicle'

                    new_actor = ActorConfigurationData(model, carla.Transform(), rolename)
                    new_actor.transform = self._get_actor_transform(rolename)

                    if ego_vehicle:
                        self.ego_vehicles.append(new_actor)
                    else:
                        self.other_actors.append(new_actor)

    def _get_actor_transform(self, actor_name):
        """
        Get the initial actor transform provided by the Init section

        Note: - The OpenScenario specification allows multiple definitions. We use the _first_ one
              - The OpenScenario specification allows different ways of specifying a position.
                We currently only support a specification with absolute world coordinates
        """

        actor_transform = carla.Transform()

        for private_action in self.init.iter("Private"):
            if private_action.attrib.get('object', None) == actor_name:
                for position in private_action.iter('Position'):
                    transform = OpenScenarioParser.convert_position_to_transform(position)
                    if transform:
                        actor_transform = transform

        return actor_transform

    def _validate_result(self):
        """
        Check that the current scenario configuration is valid
        """
        if not self.name:
            raise AttributeError("No scenario name found")

        if not self.town:
            raise AttributeError("CARLA level not defined")

        if not self.ego_vehicles:
            raise AttributeError("No ego vehicles defined in scenario")


class OpenScenarioParser(object):

    """
    Pure static class providing conversions from OpenScenario elements to ScenarioRunner elements
    """

    @staticmethod
    def convert_position_to_transform(position):
        """
        Convert an OpenScenario position into a CARLA transform

        Not supported: Road, RelativeRoad, Lane, RelativeLane as the PythonAPI currently
                       does not provide sufficient access to OpenDrive information
                       Also not supported is Route. This can be added by checking additional
                       route information
        """

        if position.find('World') is not None:
            world_pos = position.find('World')
            x = float(world_pos.attrib.get('x', 0))
            y = float(world_pos.attrib.get('y', 0))
            z = float(world_pos.attrib.get('z', 0))
            yaw = float(world_pos.attrib.get('h', 0))
            pitch = float(world_pos.attrib.get('p', 0))
            roll = float(world_pos.attrib.get('r', 0))
            return carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(yaw=yaw, pitch=pitch, roll=roll))
        elif (position.find('RelativeWorld') is not None) or (position.find('RelativeObject') is not None):
            rel_pos = position.find('RelativeWorld')
            obj = float(rel_pos.attrib.get('object'))
            obj_actor = None

            for actor in CarlaDataProvider.get_world().get_actors():
                if actor.attributes['role_name'] == obj:
                    obj_actor = actor
                    break

            if obj_actor is None:
                raise AttributeError("Object '{}' provided as position reference is not known".format(obj))

            dx = float(rel_pos.attrib.get('dx', 0))
            dy = float(rel_pos.attrib.get('dy', 0))
            dz = float(rel_pos.attrib.get('dz', 0))

            is_absolute = False
            if rel_pos.find('Orientation') is not None:
                orientation = rel_pos.find('Orientation')
                is_absolute = (orientation.attrib.get('type') == "absolute")
                dyaw = float(orientation.attrib.get('h', 0))
                dpitch = float(orientation.attrib.get('p', 0))
                droll = float(orientation.attrib.get('r', 0))

            x = obj_actor.get_location().x + dx
            y = obj_actor.get_location().y + dy
            z = obj_actor.get_location().z + dz
            yaw = obj_actor.get_transform().rotation.yaw
            pitch = obj_actor.get_transform().rotation.pitch
            roll = obj_actor.get_transform().rotation.roll
            if not is_absolute:
                yaw = yaw + dyaw
                pitch = pitch + dpitch
                roll = roll + droll
            else:
                yaw = dyaw
                pitch = dpitch
                roll = droll

            return carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(yaw=yaw, pitch=pitch, roll=roll))
        elif position.find('Road') is not None:
            raise NotImplementedError("Road positions are not yet supported")
        elif position.find('RelativeRoad') is not None:
            raise NotImplementedError("RelativeRoad positions are not yet supported")
        elif position.find('Lane') is not None:
            raise NotImplementedError("Lane positions are not yet supported")
        elif position.find('RelativeLane') is not None:
            raise NotImplementedError("RelativeLane positions are not yet supported")
        elif position.find('Route') is not None:
            raise NotImplementedError("Route positions are not yet supported")
        else:
            raise AttributeError("Unknown position")

    @staticmethod
    def convert_condition_to_atomic(condition, actor_list):
        """
        Convert an OpenScenario condition into a Behavior/Criterion atomic

        Note: Not all conditions are currently supported.
        """

        atomic = None
        condition_name = condition.attrib.get('name')

        if condition.find('ByEntity') is not None:

            trigger_actor = None    # A-priori validation ensures that this will be not None
            triggered_actor = None

            for triggering_entities in condition.find('ByEntity').iter('TriggeringEntities'):
                for entity in triggering_entities.iter('Entity'):
                    for actor in actor_list:
                        if entity.attrib.get('name', None) == actor.attributes['role_name']:
                            trigger_actor = actor
                            break

            for entity_condition in condition.find('ByEntity').iter('EntityCondition'):
                if entity_condition.find('EndOfRoad') is not None:
                    raise NotImplementedError("EndOfRoad conditions are not yet supported")
                elif entity_condition.find('Collision') is not None:
                    atomic = py_trees.meta.inverter(
                        CollisionTest(trigger_actor, terminate_on_failure=True, name=condition_name))
                elif entity_condition.find('Offroad') is not None:
                    raise NotImplementedError("Offroad conditions are not yet supported")
                elif entity_condition.find('TimeHeadway') is not None:
                    raise NotImplementedError("TimeHeadway conditions are not yet supported")
                elif entity_condition.find('TimeToCollision') is not None:
                    ttc_condition = entity_condition.find('TimeToCollision')
                    if ttc_condition.attrib.get('rule') != "less_than":
                        raise NotImplementedError(
                            "TimeToCollision condition with the given specification is not yet supported")

                    condition_value = ttc_condition.find('value')
                    condition_target = ttc_condition.find('Target')

                    if condition_target.find('Position'):
                        position = condition_target.find('Position')
                        transform = OpenScenarioParser.convert_position_to_transform(position)
                        atomic = InTimeToArrivalToLocation(triggered_actor, condition_value, condition_target.location)
                    else:
                        for actor in actor_list:
                            if ttc_condition.attrib.get('entity', None) == actor.attributes['role_name']:
                                triggered_actor = actor
                                break

                        if triggered_actor is None:
                            raise AttributeError("Cannot find actor '{}' for condition".format(
                                ttc_condition.attrib.get('entity', None)))

                        atomic = InTimeToArrivalToVehicle(trigger_actor, triggered_actor, condition_value)
                elif entity_condition.find('Acceleration') is not None:
                    raise NotImplementedError("Acceleration conditions are not yet supported")
                elif entity_condition.find('StandStill') is not None:
                    ss_condition = entity_condition.find('StandStill')
                    duration = float(ss_condition.attrib.get('duration'))
                    atomic = StandStill(trigger_actor, condition_name, duration)
                elif entity_condition.find('Speed') is not None:
                    s_condition = entity_condition.find('Speed')
                    value = float(s_condition.attrib.get('value'))
                    if s_condition.attrib.get('rule') != "greater_than":
                        raise NotImplementedError(
                            "Speed condition with the given specification is not yet supported")
                    atomic = AccelerateToVelocity(trigger_actor, value, condition_name)
                elif entity_condition.find('RelativeSpeed') is not None:
                    raise NotImplementedError("RelativeSpeed conditions are not yet supported")
                elif entity_condition.find('TraveledDistance') is not None:
                    distance_condition = entity_condition.find('TraveledDistance')
                    distance_value = float(distance_condition.attrib.get('value'))
                    atomic = DriveDistance(trigger_actor, distance_value, name=condition_name)
                elif entity_condition.find('ReachPosition') is not None:
                    rp_condition = entity_condition.find('ReachPosition')
                    distance_value = float(rp_condition.attrib.get('tolerance'))
                    position = rp_condition.find('Position')
                    transform = OpenScenarioParser.convert_position_to_transform(position)
                    atomic = InTriggerDistanceToLocation(
                        trigger_actor, transform.location, distance_value, name=condition_name)
                elif entity_condition.find('Distance') is not None:
                    distance_condition = entity_condition.find('RelativeDistance')
                    distance_value = float(distance_condition.attrib.get('value'))
                    if distance_condition.attrib.get('rule') != "less_than":
                        raise NotImplementedError(
                            "Distance condition with the given specification is not yet supported")
                    if distance_condition.find('Position'):
                        position = distance_condition.find('Position')
                        transform = OpenScenarioParser.convert_position_to_transform(position)
                        atomic = InTriggerDistanceToLocation(
                            triggered_actor, transform.location, distance_value, name=condition_name)
                elif entity_condition.find('RelativeDistance') is not None:
                    distance_condition = entity_condition.find('RelativeDistance')
                    distance_value = float(distance_condition.attrib.get('value'))
                    if (distance_condition.attrib.get('rule') == "less_than" and
                            distance_condition.attrib.get('type') == "inertial"):
                        for actor in actor_list:
                            if distance_condition.attrib.get('entity', None) == actor.attributes['role_name']:
                                triggered_actor = actor
                                break

                        if triggered_actor is None:
                            raise AttributeError("Cannot find actor '{}' for condition".format(
                                distance_condition.attrib.get('entity', None)))

                        atomic = InTriggerDistanceToVehicle(
                            triggered_actor, trigger_actor, distance_value, name=condition_name)
                    else:
                        raise NotImplementedError(
                            "RelativeDistance condition with the given specification is not yet supported")

        elif condition.find('ByState') is not None:
            raise NotImplementedError("ByState conditions are not yet supported")
        elif condition.find('ByValue') is not None:
            value_condition = condition.find('ByValue')
            if value_condition.find('Parameter') is not None:
                parameter_condition = value_condition.find('Parameter')
                arg_name = parameter_condition.attrib.get('name')
                value = parameter_condition.attrib.get('value')
                if value != '':
                    arg_value = float(value)
                else:
                    arg_value = 0
                parameter_condition.attrib.get('rule')

                if condition_name in globals():
                    criterion_instance = globals()[condition_name]
                else:
                    raise AttributeError(
                        "The condition {} cannot be mapped to a criterion atomic".format(condition_name))

                for triggered_actor in actor_list:
                    if arg_name != '':
                        atomic = criterion_instance(triggered_actor, arg_value)
                    else:
                        atomic = criterion_instance(triggered_actor)
            elif value_condition.find('SimulationTime') is not None:
                simtime_condition = value_condition.find('SimulationTime')
                value = simtime_condition.attrib.get('value')
                rule = simtime_condition.attrib.get('value')
                if rule != "greater_than":
                    raise NotImplementedError(
                        "ByValue SimulationTime conditions with the given specification is not yet supported")
                atomic = TimeOut(value)
            elif value_condition.find('TimeOfDay') is not None:
                raise NotImplementedError("ByValue TimeOfDay conditions are not yet supported")
            else:
                raise AttributeError("Unknown ByValue condition")

        else:
            raise AttributeError("Unknown condition")

        return atomic

    @staticmethod
    def convert_maneuver_to_atomic(action, actor):
        """
        Convert an OpenScenario maneuver action into a Behavior atomic

        Note not all OpenScenario actions are currently supported
        """

        maneuver_name = action.attrib.get('name', 'unknown')

        if action.find('Global') is not None:
            raise NotImplementedError("Global actions are not yet supported")
        elif action.find('UserDefined') is not None:
            user_defined_action = action.find('UserDefined')
            if user_defined_action.find('Command') is not None:
                command = user_defined_action.find('Command').text.replace(" ", "")
                if command == 'Idle':
                    atomic = Idle()
                else:
                    raise AttributeError("Unknown user command action: {}".format(command))
            else:
                raise NotImplementedError("UserDefined script actions are not yet supported")
        elif action.find('Private') is not None:
            private_action = action.find('Private')
            if private_action.find('Longitudinal') is not None:
                private_action = private_action.find('Longitudinal')
                if private_action.find('Speed') is not None:
                    long_maneuver = private_action.find('Speed')
                    target_speed = float(long_maneuver.find("Target").find("Absolute").attrib.get('value', 0))
                    distance = float(long_maneuver.find("Dynamics").attrib.get('distance', float("inf")))
                    duration = float(long_maneuver.find("Dynamics").attrib.get('time', float("inf")))
                    atomic = KeepVelocity(actor,
                                          target_speed,
                                          distance=distance,
                                          duration=duration,
                                          name=maneuver_name)
                elif private_action.find('Distance') is not None:
                    raise NotImplementedError("Longitudinal distance actions are not yet supported")
                else:
                    raise AttributeError("Unknown longitudinal action")
            elif private_action.find('Lateral') is not None:
                raise NotImplementedError("Lateral actions are not yet supported")
            elif private_action.find('Visibility') is not None:
                raise NotImplementedError("Visibility actions are not yet supported")
            elif private_action.find('Meeting') is not None:
                raise NotImplementedError("Meeting actions are not yet supported")
            elif private_action.find('Autonomous') is not None:
                private_action = private_action.find('Autonomous')
                activate = private_action.attrib.get('activate')
                atomic = ChangeAutoPilot(actor, activate, name=maneuver_name)
            elif private_action.find('Controller') is not None:
                raise NotImplementedError("Controller actions are not yet supported")
            elif private_action.find('Position') is not None:
                position = private_action.find('Position')
                transform = OpenScenarioParser.convert_position_to_transform(position)
                atomic = ActorTransformSetter(actor, transform, name=maneuver_name)
            elif private_action.find('Routing') is not None:
                private_action = private_action.find('Routing')
                if private_action.find('FollowRoute') is not None:
                    private_action = private_action.find('FollowRoute')
                    if private_action.find('Route') is not None:
                        route = private_action.find('Route')
                        plan = []
                        for waypoint in route.iter('Waypoint'):
                            position = waypoint.find('Position')
                            transform = OpenScenarioParser.convert_position_to_transform(position)
                            waypoint = CarlaDataProvider.get_map().get_waypoint(transform.location)
                            plan.append((waypoint, RoadOption.LANEFOLLOW))
                        atomic = WaypointFollower(actor, target_speed=5, plan=plan, name=maneuver_name)
                    elif private_action.find('CatalogReference') is not None:
                        raise NotImplementedError("CatalogReference private actions are not yet supported")
                    else:
                        raise AttributeError("Unknown private FollowRoute action")
                elif private_action.find('FollowTrajectory') is not None:
                    raise NotImplementedError("Private FollowTrajectory actions are not yet supported")
                elif private_action.find('AcquirePosition') is not None:
                    raise NotImplementedError("Private AcquirePosition actions are not yet supported")
                else:
                    raise AttributeError("Unknown private routing action")
            else:
                raise AttributeError("Unknown private action")

        else:
            raise AttributeError("Unknown action")

        return atomic
