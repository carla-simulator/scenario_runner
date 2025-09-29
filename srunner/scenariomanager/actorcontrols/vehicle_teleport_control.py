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
from srunner.scenariomanager.actorcontrols.basic_control import BasicControl
from srunner.scenariomanager.timer import GameTime


class VehicleTeleportControl(BasicControl):

    """
    Controller class for vehicles derived from BasicControl.

    It controls the vehicle by teleporting it through the similation.
    This only works if the given waypoints are given for all timesteps.

    Args:
        actor (carla.Actor): Vehicle actor that should be controlled.
    """


    def __init__(self, actor, args=None):
        self._prev_yaw = None
        self._distance_threshold = 0.01
        super(VehicleTeleportControl, self).__init__(actor)

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
        if not self._prev_yaw:
            self._prev_yaw = self._actor.get_transform().rotation.yaw

        transform = self._waypoints.pop(0)
        vec = self._waypoints[0].location - transform.location

        if abs(vec.x) < self._distance_threshold and abs(vec.y) < self._distance_threshold:
            yaw = self._prev_yaw  # Vehicle is stopped, avoid changing the yaw
        else:
            yaw = math.degrees(math.atan2(vec.y, vec.x))

        self._actor.set_transform(carla.Transform(transform.location, carla.Rotation(yaw=yaw)))

        self._prev_yaw = yaw
