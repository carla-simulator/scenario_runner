#!/usr/bin/env python

# Copyright (c) 2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides an example control for vehicles
"""

import math

import carla
from agents.navigation.basic_agent import LocalPlanner
from agents.navigation.local_planner import RoadOption

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.actorcontrols.basic_control import BasicControl
from srunner.scenariomanager.timer import GameTime


class VehicleVelocityControl(BasicControl):

    """
    Controller class for vehicles derived from BasicControl.

    The controller makes use of the LocalPlanner implemented in CARLA.

    Args:
        actor (carla.Actor): Vehicle actor that should be controlled.
    """

    _args = {'K_P': 1.0, 'K_D': 0.01, 'K_I': 0.0, 'dt': 0.05}

    def __init__(self, actor, args=None):

        super(VehicleVelocityControl, self).__init__(actor)

        # Remove the friction of the wheels so that the velocities aren't reduced by the ground
        physics_control = self._actor.get_physics_control()
        wheels_control = physics_control.wheels
        for w in wheels_control:
            w.tire_friction = 0
        physics_control.wheels = wheels_control
        self._actor.apply_physics_control(physics_control)

        # This is the maximum amount of time used to aim the vehicle towards the next point.
        # Used to avoid weird lateral slides if the time between point is high.
        self._max_yaw_time = 1
        self._distance_threshold = 1

    def reset(self):
        """
        Reset the controller
        """
        pass

    def run_step(self):
        """
        Execute on tick of the controller's control loop

        Teleports the vehicle to a given waypoint on each tick. Use the location from the waypoints
        but calculate the transform so that it is always pointing towards the next waypoint.
        """

        if not self._waypoints:
            return
        if not self._start_time:
            self._start_time = GameTime.get_time()

        current_transform = self._actor.get_transform()
        current_time = GameTime.get_time()
        num_remove = 0
        for t in self._times:
            if t > current_time - self._start_time:
                break
            num_remove += 1

        for _ in range(num_remove):
            self._waypoints.pop(0)
            self._times.pop(0)

        target_transform = self._waypoints[0]
        target_time = self._times[0]
        delta_time = target_time + self._start_time - current_time

        target_vec = target_transform.location - current_transform.location
        target_vec = target_vec.make_unit_vector()
        linear_speed = (target_transform.location.distance(current_transform.location)) / delta_time
        linear_velocity = linear_speed * target_vec

        self._actor.set_target_velocity(linear_velocity)

        reached_end = True
        for i in range(len(self._waypoints)):
            target_transform = self._waypoints[i]
            target_time = self._times[i]
            if current_transform.location.distance(target_transform.location) > self._distance_threshold:
                reached_end = False
                break

        if not reached_end:
            delta_time = target_time + self._start_time - current_time
            target_vec = target_transform.location - current_transform.location
            delta_yaw = math.degrees(math.atan2(target_vec.y, target_vec.x)) - current_transform.rotation.yaw
            delta_yaw = delta_yaw % 360
            if delta_yaw > 180:
                delta_yaw = delta_yaw - 360
            angular_speed = delta_yaw / min(delta_time, self._max_yaw_time)
        else:
            angular_speed = 0

        angular_velocity = carla.Vector3D(0, 0, angular_speed)

        self._actor.set_target_angular_velocity(angular_velocity)
        self._actor.apply_control(carla.VehicleControl(throttle=0.3))  # Make the wheels turn
