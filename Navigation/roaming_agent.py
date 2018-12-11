#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" This module implements an agent that roams around a track following random waypoints and avoiding other vehicles.
The agent also responds to traffic lights. """

from enum import Enum

import carla
from Navigation.local_planner import LocalPlanner
from Tools.misc import is_within_distance_ahead


class AgentState(Enum):
    """
    AGENT_STATE represents the possible states of a roaming agent
    """
    NAVIGATING = 1
    BLOCKED_BY_VEHICLE = 2
    BLOCKED_RED_LIGHT = 3


class RoamingAgent(object):
    """
    RoamingAgent implements a basic agent that navigates scenes making random choices when facing an intersection.
    This agent respects traffic lights and other vehicles.
    """

    def __init__(self, vehicle):
        """

        :param vehicle: actor to apply to local planner logic onto
        """
        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        self._map = self._vehicle.get_world().get_map()
        self._proximity_threshold = 10.0  # meters
        self._state = AgentState.NAVIGATING
        self._local_planner = LocalPlanner(self._vehicle)

    def run_step(self, debug=False):
        """
        Execute one step of navigation.
        :return:
        """

        # is there an obstacle in front of us?
        hazard_detected = False
        current_location = self._vehicle.get_location()
        vehicle_waypoint = self._map.get_waypoint(self._vehicle.get_location())

        # retrieve relevant elements for safe navigation, i.e.: traffic lights
        # and other vehicles
        actor_list = self._world.get_actors()
        vehicle_list = actor_list.filter("*vehicle*")
        lights_list = actor_list.filter("*traffic_light*")

        # check possible obstacles
        for object in vehicle_list:
            # do not account for the ego vehicle
            if object.id == self._vehicle.id:
                continue
            # if the object is not in our lane it's not an obstacle
            object_waypoint = self._map.get_waypoint(object.get_location())
            if object_waypoint.road_id != vehicle_waypoint.road_id or object_waypoint.lane_id != vehicle_waypoint.lane_id:
                continue

            loc = object.get_location()
            if is_within_distance_ahead(loc, current_location, self._vehicle.get_transform().rotation.yaw,
                                        self._proximity_threshold):
                if debug:
                    print(
                        '!!! HAZARD [{}] ==> (x={}, y={})'.format(
                            object.id, loc.x, loc.y))
                self._state = AgentState.BLOCKED_BY_VEHICLE
                hazard_detected = True
                break

        # check for the state of the traffic lights
        for object in lights_list:
            object_waypoint = self._map.get_waypoint(object.get_location())
            if object_waypoint.road_id != vehicle_waypoint.road_id or object_waypoint.lane_id != vehicle_waypoint.lane_id:
                continue

            loc = object.get_location()
            if is_within_distance_ahead(loc, current_location, self._vehicle.get_transform().rotation.yaw,
                                        self._proximity_threshold):
                if object.state == carla.libcarla.TrafficLightState.Red:
                    if debug:
                        print(
                            '=== RED LIGHT AHEAD [{}] ==> (x={}, y={})'.format(
                                object.id, loc.x, loc.y))
                    self._state = AgentState.BLOCKED_RED_LIGHT
                    hazard_detected = True
                    break

        if hazard_detected:
            self.emergency_stop()
        else:
            self._state = AgentState.NAVIGATING
            # standard local planner behavior
            self._local_planner.run_step()

    def emergency_stop(self):
        """
        Send an emergency stop command to the vehicle
        :return:
        """
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 1.0
        control.hand_brake = False
        self._vehicle.apply_control(control)
