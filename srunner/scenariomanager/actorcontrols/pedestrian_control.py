#!/usr/bin/env python

# Copyright (c) 2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides an example control for pedestrians
"""

import math

import carla

from srunner.scenariomanager.actorcontrols.basic_control import BasicControl
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider


class PedestrianControl(BasicControl):

    """
    Controller class for pedestrians derived from BasicControl.

    Args:
        actor (carla.Actor): Pedestrian actor that should be controlled.
    """

    def __init__(self, actor, args=None):
        if not isinstance(actor, carla.Walker):
            raise RuntimeError("PedestrianControl: The to be controlled actor is not a pedestrian")

        super(PedestrianControl, self).__init__(actor)

        bp = CarlaDataProvider.get_world().get_blueprint_library().find('sensor.other.collision')
        self._collision_sensor = CarlaDataProvider.get_world().spawn_actor(
                bp, carla.Transform(carla.Location(x=self._actor.bounding_box.extent.x, z=1.0)), attach_to=self._actor)
        self._collision_sensor.listen(lambda event: self._on_collision(event))  # pylint: disable=unnecessary-lambda
        self._colliding_actor = None

    def _on_collision(self, event):
        if not event:
            return
        self._colliding_actor = event.other_actor

    def reset(self):
        """
        Reset the controller
        """
        if self._actor and self._actor.is_alive:
            self._actor = None

    def run_step(self):
        """
        Execute on tick of the controller's control loop

        Note: Walkers / pedestrians are not able to walk backwards.

        If _waypoints are provided, the pedestrian moves towards the next waypoint
        with the given _target_speed, until reaching the final waypoint. Upon reaching
        the final waypoint, _reached_goal is set to True.

        If _waypoints is empty, the pedestrians moves in its current direction with
        the given _target_speed.
        """
        if not self._actor or not self._actor.is_alive:
            return

        control = self._actor.get_control()
        control.speed = self._target_speed

        # If target speed is negavite, raise an exception
        if self._target_speed < 0:
            raise NotImplementedError("Negative target speeds are not yet supported")

        if self._waypoints:
            self._reached_goal = False
            location = self._waypoints[0].location
            direction = location - self._actor.get_location()
            direction_norm = math.sqrt(direction.x**2 + direction.y**2)
            # It may happen that a pedestrian gets stuck when stepping on a sidewalk
            # Use an upwards direction to improve behavior
            if self._colliding_actor is not None and self._colliding_actor.type_id == "static.sidewalk":
                current_transform = self._actor.get_transform()
                new_transform = current_transform
                new_transform.location = new_transform.location + carla.Location(z=0.3)
                self._actor.set_transform(new_transform)
                self._colliding_actor = None
                return
                #direction = direction + carla.Location(z=0.3)
            control.direction = direction / direction_norm
            self._actor.apply_control(control)
            if direction_norm < 1.0:
                self._waypoints = self._waypoints[1:]
                if not self._waypoints:
                    self._reached_goal = True

            #for wpt in self._waypoints:
            #    begin = wpt.location + carla.Location(z=1.0)
            #    angle = math.radians(wpt.rotation.yaw)
            #    end = begin + carla.Location(x=math.cos(angle), y=math.sin(angle))
            #    CarlaDataProvider.get_world().debug.draw_arrow(begin, end, arrow_size=0.3, life_time=1.0)

        else:
            control.direction = self._actor.get_transform().rotation.get_forward_vector()
            self._actor.apply_control(control)

        self._colliding_actor = None
