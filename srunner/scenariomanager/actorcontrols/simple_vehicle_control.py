#!/usr/bin/env python

# Copyright (c) 2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides an example control for vehicles which
does not use CARLA's vehicle engine.

Limitations:
- Does not respect any traffic regulation: speed limit, traffic light, priorities, etc.
"""

import math

import carla

from srunner.scenariomanager.actorcontrols.basic_control import BasicControl
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider


class SimpleVehicleControl(BasicControl):

    """
    Controller class for vehicles derived from BasicControl.

    The controller directly sets velocities in CARLA, therefore bypassing
    CARLA's vehicle engine.

    Args:
        actor (carla.Actor): Vehicle actor that should be controlled.
    """

    def __init__(self, actor, args=None):
        super(SimpleVehicleControl, self).__init__(actor)
        self._generated_waypoint_list = []

    def reset(self):
        """
        Reset the controller
        """
        if self._actor and self._actor.is_alive:
            self._actor = None

    def run_step(self):
        """
        Execute on tick of the controller's control loop

        If _waypoints are provided, the vehicle moves towards the next waypoint
        with the given _target_speed, until reaching the final waypoint. Upon reaching
        the final waypoint, _reached_goal is set to True.

        If _waypoints is empty, the vehicle moves in its current direction with
        the given _target_speed.
        """
        self._reached_goal = False

        if not self._waypoints:
            # get next waypoint from map, to avoid leaving the road
            # then navigate to this waypoint
            self._reached_goal = False

            map_wp = None
            if not self._generated_waypoint_list:
                map_wp = CarlaDataProvider.get_map().get_waypoint(CarlaDataProvider.get_location(self._actor))
            else:
                map_wp = CarlaDataProvider.get_map().get_waypoint(self._generated_waypoint_list[-1].location)
            while len(self._generated_waypoint_list) < 50:
                map_wp = map_wp.next(2.0)[0]
                self._generated_waypoint_list.append(map_wp.transform)

            direction_norm = self._set_new_velocity(self._generated_waypoint_list[0].location)
            if direction_norm < 1.0:
                self._generated_waypoint_list = self._generated_waypoint_list[1:]
        else:
            # calculate required heading to reach next waypoint
            self._reached_goal = False
            direction_norm = self._set_new_velocity(self._waypoints[0].location)
            if direction_norm < 1.0:
                self._waypoints = self._waypoints[1:]
                if not self._waypoints:
                    self._reached_goal = True

    def _set_new_velocity(self, next_location):
        """
        Calculate and set the new actor veloctiy given the current actor
        location and the _next_location_

        Args:
            next_location (carla.Location): Next target location of the actor

        returns:
            direction (carla.Vector3D): Normalized direction vector of the actor
        """

        # set new linear velocity
        velocity = carla.Vector3D(0, 0, 0)
        direction = next_location - CarlaDataProvider.get_location(self._actor)
        direction_norm = math.sqrt(direction.x**2 + direction.y**2)
        velocity.x = direction.x / direction_norm * self._target_speed
        velocity.y = direction.y / direction_norm * self._target_speed
        self._actor.set_velocity(velocity)

        # set new angular velocity
        current_yaw = CarlaDataProvider.get_transform(self._actor).rotation.yaw
        new_yaw = CarlaDataProvider.get_map().get_waypoint(next_location).transform.rotation.yaw
        delta_yaw = new_yaw - current_yaw

        if math.fabs(delta_yaw) > 360:
            delta_yaw = delta_yaw % 360

        if delta_yaw > 180:
            delta_yaw = delta_yaw - 360
        elif delta_yaw < -180:
            delta_yaw = delta_yaw + 360

        angular_velocity = carla.Vector3D(0, 0, 0)
        angular_velocity.z = delta_yaw / (direction_norm / self._target_speed)
        self._actor.set_angular_velocity(angular_velocity)

        return direction_norm
