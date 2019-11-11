#!/usr/bin/env python

# Copyright (c) 2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides a parser for scenario configuration files based on OpenSCENARIO
"""

import math

import carla
from agents.navigation.local_planner import RoadOption

from srunner.scenariomanager.scenarioatomics.atomic_behaviors import *
from srunner.scenariomanager.scenarioatomics.atomic_criteria import *
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import *
from srunner.scenariomanager.timer import SimulationTimeCondition


class OpenScenarioParser(object):

    """
    Pure static class providing conversions from OpenSCENARIO elements to ScenarioRunner elements
    """
    operators = {
        "greater_than": operator.gt,
        "less_than": operator.lt,
        "equal_to": operator.eq
    }

    use_carla_coordinate_system = False

    @staticmethod
    def set_use_carla_coordinate_system():
        """
        CARLA internally uses a left-hand coordinate system (Unreal), but OpenSCENARIO and OpenDRIVE
        are intended for right-hand coordinate system. Hence, we need to invert the coordinates, if
        the scenario does not use CARLA coordinates, but instead right-hand coordinates.
        """
        OpenScenarioParser.use_carla_coordinate_system = True

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
            yaw = math.degrees(float(world_pos.attrib.get('h', 0)))
            pitch = math.degrees(float(world_pos.attrib.get('p', 0)))
            roll = math.degrees(float(world_pos.attrib.get('r', 0)))
            if not OpenScenarioParser.use_carla_coordinate_system:
                y = y * (-1.0)
                yaw = yaw * (-1.0)
            return carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(yaw=yaw, pitch=pitch, roll=roll))
        elif (position.find('RelativeWorld') is not None) or (position.find('RelativeObject') is not None):
            rel_pos = position.find('RelativeWorld') or position.find('RelativeObject')
            obj = rel_pos.attrib.get('object')
            obj_actor = None

            for actor in CarlaDataProvider.get_world().get_actors():
                if 'role_name' in actor.attributes and actor.attributes['role_name'] == obj:
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
                dyaw = math.degrees(float(orientation.attrib.get('h', 0)))
                dpitch = math.degrees(float(orientation.attrib.get('p', 0)))
                droll = math.degrees(float(orientation.attrib.get('r', 0)))

            if not OpenScenarioParser.use_carla_coordinate_system:
                dy = dy * (-1.0)
                dyaw = dyaw * (-1.0)

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
        Convert an OpenSCENARIO condition into a Behavior/Criterion atomic

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

                    condition_rule = ttc_condition.attrib.get('rule')
                    condition_operator = OpenScenarioParser.operators[condition_rule]

                    condition_value = ttc_condition.find('value')
                    condition_target = ttc_condition.find('Target')

                    if condition_target.find('Position'):
                        position = condition_target.find('Position')
                        transform = OpenScenarioParser.convert_position_to_transform(position)
                        atomic = InTimeToArrivalToLocation(
                            triggered_actor, condition_value, condition_target.location, condition_operator)
                    else:
                        for actor in actor_list:
                            if ttc_condition.attrib.get('entity', None) == actor.attributes['role_name']:
                                triggered_actor = actor
                                break

                        if triggered_actor is None:
                            raise AttributeError("Cannot find actor '{}' for condition".format(
                                ttc_condition.attrib.get('entity', None)))

                        atomic = InTimeToArrivalToVehicle(
                            trigger_actor, triggered_actor, condition_value, condition_operator)
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
                    distance_condition = entity_condition.find('Distance')
                    distance_value = float(distance_condition.attrib.get('value'))
                    distance_rule = distance_condition.attrib.get('rule')
                    distance_operator = OpenScenarioParser.operators[distance_rule]

                    if distance_condition.find('Position'):
                        position = distance_condition.find('Position')
                        transform = OpenScenarioParser.convert_position_to_transform(position)
                        atomic = InTriggerDistanceToLocation(
                            triggered_actor, transform.location, distance_value, distance_operator, name=condition_name)
                elif entity_condition.find('RelativeDistance') is not None:
                    distance_condition = entity_condition.find('RelativeDistance')
                    distance_value = float(distance_condition.attrib.get('value'))
                    if distance_condition.attrib.get('type') == "inertial":
                        for actor in actor_list:
                            if distance_condition.attrib.get('entity', None) == actor.attributes['role_name']:
                                triggered_actor = actor
                                break

                        if triggered_actor is None:
                            raise AttributeError("Cannot find actor '{}' for condition".format(
                                distance_condition.attrib.get('entity', None)))

                        condition_rule = distance_condition.attrib.get('rule')
                        condition_operator = OpenScenarioParser.operators[condition_rule]
                        atomic = InTriggerDistanceToVehicle(
                            triggered_actor, trigger_actor, distance_value, condition_operator, name=condition_name)
                    else:
                        raise NotImplementedError(
                            "RelativeDistance condition with the given specification is not yet supported")

        elif condition.find('ByState') is not None:
            state_condition = condition.find('ByState')
            if state_condition.find('AtStart') is not None:
                element_type = state_condition.find('AtStart').attrib.get('type')
                element_name = state_condition.find('AtStart').attrib.get('name')
                atomic = AtStartCondition(element_type, element_name)
            elif state_condition.find('AfterTermination') is not None:
                element_type = state_condition.find('AfterTermination').attrib.get('type')
                element_name = state_condition.find('AfterTermination').attrib.get('name')
                condition_rule = state_condition.find('AfterTermination').attrib.get('rule')
                atomic = AfterTerminationCondition(element_type, element_name, condition_rule)
            elif state_condition.find('Command') is not None:
                raise NotImplementedError("ByState Command conditions are not yet supported")
            elif state_condition.find('Signal') is not None:
                raise NotImplementedError("ByState Signal conditions are not yet supported")
            elif state_condition.find('Controller') is not None:
                raise NotImplementedError("ByState Controller conditions are not yet supported")
            else:
                raise AttributeError("Unknown ByState condition")
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
                value = float(simtime_condition.attrib.get('value'))
                rule = simtime_condition.attrib.get('rule')
                atomic = SimulationTimeCondition(value, success_rule=rule)
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
        Convert an OpenSCENARIO maneuver action into a Behavior atomic

        Note not all OpenSCENARIO actions are currently supported
        """

        maneuver_name = action.attrib.get('name', 'unknown')

        if action.find('Global') is not None:
            global_action = action.find('Global')
            if global_action.find('Infrastructure') is not None:
                infrastructure_action = global_action.find('Infrastructure').find('Signal')
                if infrastructure_action.find('SetState') is not None:
                    traffic_light_action = infrastructure_action.find('SetState')
                    traffic_light_id = traffic_light_action.attrib.get('name')
                    traffic_light_state = traffic_light_action.attrib.get('state')
                    atomic = TrafficLightStateSetter(traffic_light_id, traffic_light_state)
                else:
                    raise NotImplementedError("TrafficLights can only be influenced via SetState")
            else:
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
                private_action = private_action.find('Lateral')
                if private_action.find('LaneChange') is not None:
                    lat_maneuver = private_action.find('LaneChange')
                    target_lane_rel = float(lat_maneuver.find("Target").find("Relative").attrib.get('value', 0))
                    distance = float(lat_maneuver.find("Dynamics").attrib.get('distance', float("inf")))
                    atomic = LaneChange(actor,
                                        None,
                                        direction="left" if target_lane_rel < 0 else "right",
                                        distance_lane_change=distance,
                                        name=maneuver_name)
                else:
                    raise AttributeError("Unknown lateral action")
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
                target_speed = 5.0
                private_action = private_action.find('Routing')
                if private_action.find('FollowRoute') is not None:
                    private_action = private_action.find('FollowRoute')
                    if private_action.find('Route') is not None:
                        route = private_action.find('Route')
                        plan = []
                        if route.find('ParameterDeclaration') is not None:
                            if route.find('ParameterDeclaration').find('Parameter') is not None:
                                parameter = route.find('ParameterDeclaration').find('Parameter')
                                if parameter.attrib.get('name') == "Speed":
                                    target_speed = float(parameter.attrib.get('value', 5.0))
                        for waypoint in route.iter('Waypoint'):
                            position = waypoint.find('Position')
                            transform = OpenScenarioParser.convert_position_to_transform(position)
                            waypoint = CarlaDataProvider.get_map().get_waypoint(transform.location)
                            plan.append((waypoint, RoadOption.LANEFOLLOW))
                        atomic = WaypointFollower(actor, target_speed=target_speed, plan=plan, name=maneuver_name)
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
