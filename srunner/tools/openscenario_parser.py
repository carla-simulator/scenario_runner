#!/usr/bin/env python

# Copyright (c) 2019-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides a parser for scenario configuration files based on OpenSCENARIO
"""

from distutils.util import strtobool
import datetime
import math
import operator

import py_trees
import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.weather_sim import Weather
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (TrafficLightStateSetter,
                                                                      ActorTransformSetterToOSCPosition,
                                                                      AccelerateToVelocity,
                                                                      ChangeAutoPilot,
                                                                      KeepVelocity,
                                                                      LaneChange,
                                                                      RunScript,
                                                                      SetRelativeOSCVelocity,
                                                                      UpdateWeather,
                                                                      UpdateRoadFriction,
                                                                      Idle,
                                                                      WaypointFollower)
# pylint: disable=unused-import
# For the following includes the pylint check is disabled, as these are accessed via globals()
from srunner.scenariomanager.scenarioatomics.atomic_criteria import (CollisionTest,
                                                                     MaxVelocityTest,
                                                                     DrivenDistanceTest,
                                                                     AverageVelocityTest,
                                                                     KeepLaneTest,
                                                                     ReachedRegionTest,
                                                                     OnSidewalkTest,
                                                                     WrongLaneTest,
                                                                     InRadiusRegionTest,
                                                                     InRouteTest,
                                                                     RouteCompletionTest,
                                                                     RunningRedLightTest,
                                                                     RunningStopTest)
# pylint: enable=unused-import
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (InTriggerDistanceToVehicle,
                                                                               InTriggerDistanceToOSCPosition,
                                                                               InTimeToArrivalToOSCPosition,
                                                                               InTimeToArrivalToVehicle,
                                                                               DriveDistance,
                                                                               StandStill,
                                                                               OSCStartEndCondition)
from srunner.scenariomanager.timer import TimeOut, SimulationTimeCondition
from srunner.tools.py_trees_port import oneshot_behavior


class OpenScenarioParser(object):

    """
    Pure static class providing conversions from OpenSCENARIO elements to ScenarioRunner elements
    """
    operators = {
        "greaterThan": operator.gt,
        "lessThan": operator.lt,
        "equalTo": operator.eq
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
    def get_friction_from_env_action(xml_tree, catalogs):
        """
        Extract the CARLA road friction coefficient from an OSC EnvironmentAction

        Args:
            xml_tree: Containing the EnvironmentAction,
                or the reference to the catalog it is defined in.
            catalogs: XML Catalogs that could contain the EnvironmentAction

        returns:
           friction (float)
        """
        set_environment = next(xml_tree.iter("EnvironmentAction"))

        friction = 1.0

        road_condition = set_environment.iter("RoadCondition")
        for condition in road_condition:
            friction = condition.attrib.get('frictionScaleFactor')

        return friction

    @staticmethod
    def get_weather_from_env_action(xml_tree, catalogs):
        """
        Extract the CARLA weather parameters from an OSC EnvironmentAction

        Args:
            xml_tree: Containing the EnvironmentAction,
                or the reference to the catalog it is defined in.
            catalogs: XML Catalogs that could contain the EnvironmentAction

        returns:
           Weather (srunner.scenariomanager.weather_sim.Weather)
        """
        set_environment = next(xml_tree.iter("EnvironmentAction"))

        if sum(1 for _ in set_environment.iter("Weather")) != 0:
            environment = set_environment.find("Environment")
        elif set_environment.find("CatalogReference") is not None:
            catalog_reference = set_environment.find("CatalogReference")
            environment = catalogs[catalog_reference.attrib.get(
                "catalogName")][catalog_reference.attrib.get("entryName")]

        weather = environment.find("Weather")
        sun = weather.find("Sun")

        carla_weather = carla.WeatherParameters()
        carla_weather.sun_azimuth_angle = math.degrees(float(sun.attrib.get('azimuth', 0)))
        carla_weather.sun_altitude_angle = math.degrees(float(sun.attrib.get('elevation', 0)))
        carla_weather.cloudiness = 100 - float(sun.attrib.get('intensity', 0)) * 100
        fog = weather.find("Fog")
        carla_weather.fog_distance = float(fog.attrib.get('visualRange', 'inf'))
        if carla_weather.fog_distance < 1000:
            carla_weather.fog_density = 100
        carla_weather.precipitation = 0
        carla_weather.precipitation_deposits = 0
        carla_weather.wetness = 0
        carla_weather.wind_intensity = 0
        precepitation = weather.find("Precipitation")
        if precepitation.attrib.get('precipitationType') == "rain":
            carla_weather.precipitation = float(precepitation.attrib.get('intensity')) * 100
            carla_weather.precipitation_deposits = 100  # if it rains, make the road wet
            carla_weather.wetness = carla_weather.precipitation
        elif precepitation.attrib.get('type') == "snow":
            raise AttributeError("CARLA does not support snow precipitation")

        time_of_day = environment.find("TimeOfDay")
        weather_animation = strtobool(time_of_day.attrib.get("animation"))
        time = time_of_day.attrib.get("dateTime")
        dtime = datetime.datetime.strptime(time, "%Y-%m-%dT%H:%M:%S")

        return Weather(carla_weather, dtime, weather_animation)

    @staticmethod
    def convert_position_to_transform(position, actor_list=None):
        """
        Convert an OpenScenario position into a CARLA transform

        Not supported: Road, RelativeRoad, Lane, RelativeLane as the PythonAPI currently
                       does not provide sufficient access to OpenDrive information
                       Also not supported is Route. This can be added by checking additional
                       route information
        """

        if position.find('WorldPosition') is not None:
            world_pos = position.find('WorldPosition')
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

        elif ((position.find('RelativeWorldPosition') is not None) or
              (position.find('RelativeObjectPosition') is not None) or
              (position.find('RelativeLanePosition') is not None)):
            rel_pos = position.find('RelativeWorldPosition') or position.find(
                'RelativeObjectPosition') or position.find('RelativeLanePosition')

            # get relative object and relative position
            obj = rel_pos.attrib.get('object')
            obj_actor = None
            actor_transform = None

            if actor_list is not None:
                for actor in actor_list:
                    if actor.rolename == obj:
                        obj_actor = actor
                        actor_transform = actor.transform
            else:
                for actor in CarlaDataProvider.get_world().get_actors():
                    if 'role_name' in actor.attributes and actor.attributes['role_name'] == obj:
                        obj_actor = actor
                        actor_transform = obj_actor.get_transform()
                        break

            if obj_actor is None:
                raise AttributeError("Object '{}' provided as position reference is not known".format(obj))

            # calculate orientation h, p, r
            is_absolute = False
            if rel_pos.find('Orientation') is not None:
                orientation = rel_pos.find('Orientation')
                is_absolute = (orientation.attrib.get('type') == "absolute")
                dyaw = math.degrees(float(orientation.attrib.get('h', 0)))
                dpitch = math.degrees(float(orientation.attrib.get('p', 0)))
                droll = math.degrees(float(orientation.attrib.get('r', 0)))

            if not OpenScenarioParser.use_carla_coordinate_system:
                dyaw = dyaw * (-1.0)

            yaw = actor_transform.rotation.yaw
            pitch = actor_transform.rotation.pitch
            roll = actor_transform.rotation.roll

            if not is_absolute:
                yaw = yaw + dyaw
                pitch = pitch + dpitch
                roll = roll + droll
            else:
                yaw = dyaw
                pitch = dpitch
                roll = droll

            # calculate location x, y, z
            # dx, dy, dz
            if ((position.find('RelativeWorldPosition') is not None) or
                    (position.find('RelativeObjectPosition') is not None)):
                dx = float(rel_pos.attrib.get('dx', 0))
                dy = float(rel_pos.attrib.get('dy', 0))
                dz = float(rel_pos.attrib.get('dz', 0))

                if not OpenScenarioParser.use_carla_coordinate_system:
                    dy = dy * (-1.0)

                x = actor_transform.location.x + dx
                y = actor_transform.location.y + dy
                z = actor_transform.location.z + dz

            # dLane, ds, offset
            elif position.find('RelativeLanePosition') is not None:
                dlane = float(rel_pos.attrib.get('dLane'))
                ds = float(rel_pos.attrib.get('ds'))
                offset = float(rel_pos.attrib.get('offset'))

                carla_map = CarlaDataProvider.get_map()
                relative_waypoint = carla_map.get_waypoint(actor_transform.location)

                if dlane == 0:
                    wp = relative_waypoint
                elif dlane == -1:
                    wp = relative_waypoint.get_left_lane()
                elif dlane == 1:
                    wp = relative_waypoint.get_right_lane()
                if wp is None:
                    raise AttributeError("Object '{}' position with dLane={} is not valid".format(obj, dlane))

                wp = wp.next(ds)[-1]

                # Adapt transform according to offset
                h = math.radians(wp.transform.rotation.yaw)
                x_offset = math.sin(h) * offset
                y_offset = math.cos(h) * offset

                if OpenScenarioParser.use_carla_coordinate_system:
                    x_offset = x_offset * (-1.0)
                    y_offset = y_offset * (-1.0)

                x = wp.transform.location.x + x_offset
                y = wp.transform.location.y + y_offset
                z = wp.transform.location.z

            return carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(yaw=yaw, pitch=pitch, roll=roll))

        # Not implemented
        elif position.find('RoadPosition') is not None:
            raise NotImplementedError("Road positions are not yet supported")
        elif position.find('RelativeRoadPosition') is not None:
            raise NotImplementedError("RelativeRoad positions are not yet supported")
        elif position.find('LanePosition') is not None:
            lane_pos = position.find('LanePosition')
            road_id = int(lane_pos.attrib.get('roadId', 0))
            lane_id = int(lane_pos.attrib.get('laneId', 0))
            offset = float(lane_pos.attrib.get('offset', 0))
            s = float(lane_pos.attrib.get('s', 0))
            is_absolute = True
            waypoint = CarlaDataProvider.get_map().get_waypoint_xodr(road_id, lane_id, s)
            if waypoint is None:
                raise AttributeError("Lane position cannot be found")

            transform = waypoint.transform
            if lane_pos.find('Orientation') is not None:
                orientation = rel_pos.find('Orientation')
                dyaw = math.degrees(float(orientation.attrib.get('h', 0)))
                dpitch = math.degrees(float(orientation.attrib.get('p', 0)))
                droll = math.degrees(float(orientation.attrib.get('r', 0)))

                if not OpenScenarioParser.use_carla_coordinate_system:
                    dyaw = dyaw * (-1.0)

                transform.rotation.yaw = transform.rotation.yaw + dyaw
                transform.rotation.pitch = transform.rotation.pitch + dpitch
                transform.rotation.roll = transform.rotation.roll + droll

            if offset != 0:
                forward_vector = transform.rotation.get_forward_vector()
                orthogonal_vector = carla.Vector3D(x=-forward_vector.y, y=forward_vector.x, z=forward_vector.z)
                transform.location.x = transform.location.x + offset * orthogonal_vector.x
                transform.location.y = transform.location.y + offset * orthogonal_vector.y

            return transform
        elif position.find('RoutePosition') is not None:
            raise NotImplementedError("Route positions are not yet supported")
        else:
            raise AttributeError("Unknown position")

    @staticmethod
    def convert_condition_to_atomic(condition, actor_list):
        """
        Convert an OpenSCENARIO condition into a Behavior/Criterion atomic

        If there is a delay defined in the condition, then the condition is checked after the delay time
        passed by, e.g. <Condition name="" delay="5">.

        Note: Not all conditions are currently supported.
        """

        atomic = None
        delay_atomic = None
        condition_name = condition.attrib.get('name')

        if condition.attrib.get('delay') is not None and str(condition.attrib.get('delay')) != '0':
            delay = float(condition.attrib.get('delay'))
            delay_atomic = TimeOut(delay)

        if condition.find('ByEntityCondition') is not None:

            trigger_actor = None    # A-priori validation ensures that this will be not None
            triggered_actor = None

            for triggering_entities in condition.find('ByEntityCondition').iter('TriggeringEntities'):
                for entity in triggering_entities.iter('EntityRef'):
                    for actor in actor_list:
                        if entity.attrib.get('entityRef', None) == actor.attributes['role_name']:
                            trigger_actor = actor
                            break

            for entity_condition in condition.find('ByEntityCondition').iter('EntityCondition'):
                if entity_condition.find('EndOfRoadCondition') is not None:
                    raise NotImplementedError("EndOfRoad conditions are not yet supported")
                elif entity_condition.find('CollisionCondition') is not None:
                    atomic = py_trees.meta.inverter(
                        CollisionTest(trigger_actor, terminate_on_failure=True, name=condition_name))
                elif entity_condition.find('OffroadCondition') is not None:
                    raise NotImplementedError("Offroad conditions are not yet supported")
                elif entity_condition.find('TimeHeadwayCondition') is not None:
                    raise NotImplementedError("TimeHeadway conditions are not yet supported")
                elif entity_condition.find('TimeToCollisionCondition') is not None:
                    ttc_condition = entity_condition.find('TimeToCollisionCondition')

                    condition_rule = ttc_condition.attrib.get('rule')
                    condition_operator = OpenScenarioParser.operators[condition_rule]

                    condition_value = ttc_condition.attrib.get('value')
                    condition_target = ttc_condition.find('TimeToCollisionConditionTarget')

                    if condition_target.find('Position') is not None:
                        position = condition_target.find('Position')
                        atomic = InTimeToArrivalToOSCPosition(
                            trigger_actor, position, condition_value, condition_operator)
                    else:
                        for actor in actor_list:
                            if ttc_condition.attrib.get('EntityRef', None) == actor.attributes['role_name']:
                                triggered_actor = actor
                                break
                        if triggered_actor is None:
                            raise AttributeError("Cannot find actor '{}' for condition".format(
                                ttc_condition.attrib.get('EntityRef', None)))

                        atomic = InTimeToArrivalToVehicle(
                            trigger_actor, triggered_actor, condition_value, condition_operator)
                elif entity_condition.find('AccelerationCondition') is not None:
                    raise NotImplementedError("Acceleration conditions are not yet supported")
                elif entity_condition.find('StandStillCondition') is not None:
                    ss_condition = entity_condition.find('StandStillCondition')
                    duration = float(ss_condition.attrib.get('duration'))
                    atomic = StandStill(trigger_actor, condition_name, duration)
                elif entity_condition.find('SpeedCondition') is not None:
                    s_condition = entity_condition.find('SpeedCondition')
                    value = float(s_condition.attrib.get('value'))
                    if s_condition.attrib.get('rule') != "greaterThan":
                        raise NotImplementedError(
                            "Speed condition with the given specification is not yet supported")
                    atomic = AccelerateToVelocity(trigger_actor, value, condition_name)
                elif entity_condition.find('RelativeSpeedCondition') is not None:
                    raise NotImplementedError("RelativeSpeed conditions are not yet supported")
                elif entity_condition.find('TraveledDistanceCondition') is not None:
                    distance_condition = entity_condition.find('TraveledDistanceCondition')
                    distance_value = float(distance_condition.attrib.get('value'))
                    atomic = DriveDistance(trigger_actor, distance_value, name=condition_name)
                elif entity_condition.find('ReachPositionCondition') is not None:
                    rp_condition = entity_condition.find('ReachPositionCondition')
                    distance_value = float(rp_condition.attrib.get('tolerance'))
                    position = rp_condition.find('Position')
                    atomic = InTriggerDistanceToOSCPosition(
                        trigger_actor, position, distance_value, name=condition_name)
                elif entity_condition.find('DistanceCondition') is not None:
                    distance_condition = entity_condition.find('DistanceCondition')
                    distance_value = float(distance_condition.attrib.get('value'))
                    distance_rule = distance_condition.attrib.get('rule')
                    distance_operator = OpenScenarioParser.operators[distance_rule]
                    if distance_condition.find('Position') is not None:
                        position = distance_condition.find('Position')
                        atomic = InTriggerDistanceToOSCPosition(
                            trigger_actor, position, distance_value, distance_operator, name=condition_name)

                elif entity_condition.find('RelativeDistanceCondition') is not None:
                    distance_condition = entity_condition.find('RelativeDistanceCondition')
                    distance_value = float(distance_condition.attrib.get('value'))
                    if distance_condition.attrib.get('relativeDistanceType') == "cartesianDistance":
                        for actor in actor_list:
                            if distance_condition.attrib.get('entityRef', None) == actor.attributes['role_name']:
                                triggered_actor = actor
                                break

                        if triggered_actor is None:
                            raise AttributeError("Cannot find actor '{}' for condition".format(
                                distance_condition.attrib.get('entityRef', None)))

                        condition_rule = distance_condition.attrib.get('rule')
                        condition_operator = OpenScenarioParser.operators[condition_rule]
                        atomic = InTriggerDistanceToVehicle(
                            triggered_actor, trigger_actor, distance_value, condition_operator, name=condition_name)
                    else:
                        raise NotImplementedError(
                            "RelativeDistance condition with the given specification is not yet supported")
        elif condition.find('ByValueCondition') is not None:
            value_condition = condition.find('ByValueCondition')
            if value_condition.find('ParameterCondition') is not None:
                parameter_condition = value_condition.find('ParameterCondition')
                arg_name = parameter_condition.attrib.get('parameterRef')
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

                atomic = py_trees.composites.Parallel("Evaluation Criteria for multiple ego vehicles")
                for triggered_actor in actor_list:
                    if arg_name != '':
                        atomic.add_child(criterion_instance(triggered_actor, arg_value))
                    else:
                        atomic.add_child(criterion_instance(triggered_actor))
            elif value_condition.find('SimulationTimeCondition') is not None:
                simtime_condition = value_condition.find('SimulationTimeCondition')
                value = float(simtime_condition.attrib.get('value'))
                rule = simtime_condition.attrib.get('rule')
                atomic = SimulationTimeCondition(value, success_rule=rule)
            elif value_condition.find('TimeOfDayCondition') is not None:
                raise NotImplementedError("ByValue TimeOfDay conditions are not yet supported")
            elif value_condition.find('StoryboardElementStateCondition') is not None:
                state_condition = value_condition.find('StoryboardElementStateCondition')
                element_name = state_condition.attrib.get('storyboardElementRef')
                element_type = state_condition.attrib.get('storyboardElementType')
                state = state_condition.attrib.get('state')
                if state == "startTransition":
                    atomic = OSCStartEndCondition(element_type, element_name, rule="START", name=state + "Condition")
                elif state == "stopTransition" or state == "endTransition" or state == "completeState":
                    atomic = OSCStartEndCondition(element_type, element_name, rule="END", name=state + "Condition")
                else:
                    raise NotImplementedError(
                        "Only start, stop, endTransitions and completeState are currently supported")
            elif value_condition.find('UserDefinedValueCondition') is not None:
                raise NotImplementedError("ByValue UserDefinedValue conditions are not yet supported")
            elif value_condition.find('TrafficSignalCondition') is not None:
                raise NotImplementedError("ByValue TrafficSignal conditions are not yet supported")
            elif value_condition.find('TrafficSignalControllerCondition') is not None:
                raise NotImplementedError("ByValue TrafficSignalController conditions are not yet supported")
            else:
                raise AttributeError("Unknown ByValue condition")

        else:
            raise AttributeError("Unknown condition")

        if delay_atomic is not None and atomic is not None:
            new_atomic = py_trees.composites.Sequence("delayed sequence")
            new_atomic.add_child(delay_atomic)
            new_atomic.add_child(atomic)
        else:
            new_atomic = atomic

        return new_atomic

    @staticmethod
    def convert_maneuver_to_atomic(action, actor, catalogs):
        """
        Convert an OpenSCENARIO maneuver action into a Behavior atomic

        Note not all OpenSCENARIO actions are currently supported
        """
        maneuver_name = action.attrib.get('name', 'unknown')

        if action.find('GlobalAction') is not None:
            global_action = action.find('GlobalAction')
            if global_action.find('InfrastructureAction') is not None:
                infrastructure_action = global_action.find('InfrastructureAction').find('TrafficSignalAction')
                if infrastructure_action.find('TrafficSignalStateAction') is not None:
                    traffic_light_id = None
                    traffic_light_action = infrastructure_action.find('TrafficSignalStateAction')
                    name = traffic_light_action.attrib.get('name')
                    if name.startswith("id="):
                        traffic_light_id = name[3:]
                    elif name.startswith("pos="):
                        position = name[4:]
                        pos = position.split(",")
                        for carla_actor in CarlaDataProvider.get_world().get_actors().filter('traffic.traffic_light'):
                            carla_actor_loc = carla_actor.get_transform().location
                            distance = carla_actor_loc.distance(carla.Location(x=float(pos[0]),
                                                                               y=float(pos[1]),
                                                                               z=carla_actor_loc.z))
                            if distance < 2.0:
                                traffic_light_id = carla_actor.id
                                break
                    if traffic_light_id is None:
                        raise AttributeError("Unknown  traffic light {}".format(name))
                    traffic_light_state = traffic_light_action.attrib.get('state')
                    atomic = TrafficLightStateSetter(
                        traffic_light_id, traffic_light_state, name=maneuver_name + "_" + str(traffic_light_id))
                else:
                    raise NotImplementedError("TrafficLights can only be influenced via TrafficSignalStateAction")
            elif global_action.find('EnvironmentAction') is not None:
                weather_behavior = UpdateWeather(
                    OpenScenarioParser.get_weather_from_env_action(global_action, catalogs))
                friction_behavior = UpdateRoadFriction(
                    OpenScenarioParser.get_friction_from_env_action(global_action, catalogs))

                env_behavior = py_trees.composites.Parallel(
                    policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL, name=maneuver_name)

                env_behavior.add_child(
                    oneshot_behavior(variable_name=maneuver_name + ">WeatherUpdate", behaviour=weather_behavior))
                env_behavior.add_child(
                    oneshot_behavior(variable_name=maneuver_name + ">FrictionUpdate", behaviour=friction_behavior))

                return env_behavior

            else:
                raise NotImplementedError("Global actions are not yet supported")
        elif action.find('UserDefinedAction') is not None:
            user_defined_action = action.find('UserDefinedAction')
            if user_defined_action.find('CustomCommandAction') is not None:
                command = user_defined_action.find('CustomCommandAction').attrib.get('type')
                atomic = RunScript(command, name=maneuver_name)
        elif action.find('PrivateAction') is not None:
            private_action = action.find('PrivateAction')

            if private_action.find('LongitudinalAction') is not None:
                private_action = private_action.find('LongitudinalAction')

                if private_action.find('SpeedAction') is not None:
                    long_maneuver = private_action.find('SpeedAction')

                    # duration and distance
                    distance = float('inf')
                    duration = float('inf')
                    dimension = long_maneuver.find("SpeedActionDynamics").attrib.get('dynamicsDimension')
                    if dimension == "distance":
                        distance = float(long_maneuver.find("SpeedActionDynamics").attrib.get('value', float("inf")))
                    else:
                        duration = float(long_maneuver.find("SpeedActionDynamics").attrib.get('value', float("inf")))

                    # absolute velocity with given target speed
                    if long_maneuver.find("SpeedActionTarget").find("AbsoluteTargetSpeed") is not None:
                        target_speed = float(long_maneuver.find("SpeedActionTarget").find(
                            "AbsoluteTargetSpeed").attrib.get('value', 0))
                        atomic = KeepVelocity(actor,
                                              target_speed,
                                              distance=distance,
                                              duration=duration,
                                              name=maneuver_name)

                    # relative velocity to given actor
                    if long_maneuver.find("SpeedActionTarget").find("RelativeTargetSpeed") is not None:
                        relative_speed = long_maneuver.find("SpeedActionTarget").find("RelativeTargetSpeed")
                        obj = relative_speed.attrib.get('entityRef')
                        value = float(relative_speed.attrib.get('value', 0))
                        value_type = relative_speed.attrib.get('speedTargetValueType')
                        continuous = relative_speed.attrib.get('continuous')

                        for traffic_actor in CarlaDataProvider.get_world().get_actors():
                            if 'role_name' in traffic_actor.attributes and traffic_actor.attributes['role_name'] == obj:
                                obj_actor = traffic_actor
                        atomic = SetRelativeOSCVelocity(actor,
                                                        obj_actor,
                                                        value,
                                                        value_type,
                                                        continuous,
                                                        duration,
                                                        distance)

                elif private_action.find('LongitudinalDistanceAction') is not None:
                    raise NotImplementedError("Longitudinal distance actions are not yet supported")
                else:
                    raise AttributeError("Unknown longitudinal action")
            elif private_action.find('LateralAction') is not None:
                private_action = private_action.find('LateralAction')
                if private_action.find('LaneChangeAction') is not None:
                    # Note: LaneChangeActions are currently only supported for RelativeTargetLane
                    #       with +1 or -1 referring to the action actor
                    lat_maneuver = private_action.find('LaneChangeAction')
                    target_lane_rel = float(lat_maneuver.find("LaneChangeTarget").find(
                        "RelativeTargetLane").attrib.get('value', 0))
                    # duration and distance
                    distance = float('inf')
                    duration = float('inf')
                    dimension = lat_maneuver.find("LaneChangeActionDynamics").attrib.get('dynamicsDimension')
                    if dimension == "distance":
                        distance = float(
                            lat_maneuver.find("LaneChangeActionDynamics").attrib.get('value', float("inf")))
                    else:
                        duration = float(
                            lat_maneuver.find("LaneChangeActionDynamics").attrib.get('value', float("inf")))
                    atomic = LaneChange(actor,
                                        None,
                                        direction="left" if target_lane_rel < 0 else "right",
                                        distance_lane_change=distance,
                                        name=maneuver_name)
                else:
                    raise AttributeError("Unknown lateral action")
            elif private_action.find('VisibilityAction') is not None:
                raise NotImplementedError("Visibility actions are not yet supported")
            elif private_action.find('SynchronizeAction') is not None:
                raise NotImplementedError("Synchronization actions are not yet supported")
            elif private_action.find('ActivateControllerAction') is not None:
                private_action = private_action.find('ActivateControllerAction')
                activate = strtobool(private_action.attrib.get('longitudinal'))
                atomic = ChangeAutoPilot(actor, activate, name=maneuver_name)
            elif private_action.find('ControllerAction') is not None:
                raise NotImplementedError("Controller actions are not yet supported")
            elif private_action.find('TeleportAction') is not None:
                position = private_action.find('TeleportAction')
                atomic = ActorTransformSetterToOSCPosition(actor, position, name=maneuver_name)
            elif private_action.find('RoutingAction') is not None:
                target_speed = 5.0
                private_action = private_action.find('RoutingAction')
                if private_action.find('AssignRouteAction') is not None:
                    private_action = private_action.find('AssignRouteAction')
                    if private_action.find('Route') is not None:
                        route = private_action.find('Route')
                        plan = []
                        if route.find('ParameterDeclarations') is not None:
                            if route.find('ParameterDeclarations').find('Parameter') is not None:
                                parameter = route.find('ParameterDeclarations').find('Parameter')
                                if parameter.attrib.get('name') == "Speed":
                                    target_speed = float(parameter.attrib.get('value', 5.0))
                        for waypoint in route.iter('Waypoint'):
                            position = waypoint.find('Position')
                            transform = OpenScenarioParser.convert_position_to_transform(position)
                            plan.append(transform.location)
                        atomic = WaypointFollower(actor, target_speed=target_speed, plan=plan, name=maneuver_name)
                    elif private_action.find('CatalogReference') is not None:
                        raise NotImplementedError("CatalogReference private actions are not yet supported")
                    else:
                        raise AttributeError("Unknown private FollowRoute action")
                elif private_action.find('FollowTrajectoryAction') is not None:
                    raise NotImplementedError("Private FollowTrajectory actions are not yet supported")
                elif private_action.find('AcquirePositionAction') is not None:
                    raise NotImplementedError("Private AcquirePosition actions are not yet supported")
                else:
                    raise AttributeError("Unknown private routing action")
            else:
                raise AttributeError("Unknown private action")

        else:
            if action:
                raise AttributeError("Unknown action: {}".format(maneuver_name))
            else:
                return Idle(duration=0, name=maneuver_name)

        return atomic
