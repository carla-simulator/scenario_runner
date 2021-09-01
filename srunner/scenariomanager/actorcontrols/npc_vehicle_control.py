#!/usr/bin/env python

# Copyright (c) 2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides an example control for vehicles
"""

from distutils.util import strtobool
import math

import carla
from agents.navigation.basic_agent import LocalPlanner
from agents.navigation.local_planner import RoadOption

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.actorcontrols.basic_control import BasicControl


class NpcVehicleControl(BasicControl):

    """
    Controller class for vehicles derived from BasicControl.

    The controller makes use of the LocalPlanner implemented in CARLA.

    Args:
        actor (carla.Actor): Vehicle actor that should be controlled.
    """

    _args = {'K_P': 1.0, 'K_D': 0.01, 'K_I': 0.0, 'dt': 0.05}

    def __init__(self, actor, args=None):
        super(NpcVehicleControl, self).__init__(actor)

        self._local_planner = LocalPlanner(  # pylint: disable=undefined-variable
            self._actor, opt_dict={
                'target_speed': self._target_speed * 3.6,
                'lateral_control_dict': self._args})

        if self._waypoints:
            self._update_plan()

        self._obstacle_sensor = None
        self._obstacle_distance = float('inf')
        self._obstacle_actor = None
        self._consider_obstacles = False
        self._proximity_threshold = float('inf')
        self._brake_lights_active = False

        if args and 'consider_obstacles' in args and strtobool(args['consider_obstacles']):
            self._consider_obstacles = strtobool(args['consider_obstacles'])
            bp = CarlaDataProvider.get_world().get_blueprint_library().find('sensor.other.obstacle')
            bp.set_attribute('distance', '250')
            if args and 'proximity_threshold' in args:
                self._proximity_threshold = float(args['proximity_threshold'])
                bp.set_attribute('distance', str(max(float(args['proximity_threshold']), 250)))
            bp.set_attribute('hit_radius', '1')
            bp.set_attribute('only_dynamics', 'True')
            self._obstacle_sensor = CarlaDataProvider.get_world().spawn_actor(
                bp, carla.Transform(carla.Location(x=self._actor.bounding_box.extent.x, z=1.0)), attach_to=self._actor)
            self._obstacle_sensor.listen(lambda event: self._on_obstacle(event))  # pylint: disable=unnecessary-lambda

    def _on_obstacle(self, event):
        """
        Callback for the obstacle sensor

        Sets _obstacle_distance and _obstacle_actor according to the closest obstacle
        found by the sensor.
        """
        if not event:
            return
        self._obstacle_distance = event.distance
        self._obstacle_actor = event.other_actor

    def _update_plan(self):
        """
        Update the plan (waypoint list) of the LocalPlanner
        """
        plan = []
        for transform in self._waypoints:
            waypoint = CarlaDataProvider.get_map().get_waypoint(
                transform.location, project_to_road=True, lane_type=carla.LaneType.Any)
            plan.append((waypoint, RoadOption.LANEFOLLOW))
        self._local_planner.set_global_plan(plan)

    def _update_offset(self):
        """
        Update the plan (waypoint list) of the LocalPlanner
        """
        self._local_planner._vehicle_controller._lat_controller._offset = self._offset   # pylint: disable=protected-access

    def reset(self):
        """
        Reset the controller
        """
        if self._actor and self._actor.is_alive:
            if self._local_planner:
                self._local_planner.reset_vehicle()
                self._local_planner = None
            self._actor = None

    def run_step(self):
        """
        Execute on tick of the controller's control loop

        Note: Negative target speeds are not yet supported.
              Try using simple_vehicle_control or vehicle_longitudinal_control.

        If _waypoints are provided, the vehicle moves towards the next waypoint
        with the given _target_speed, until reaching the final waypoint. Upon reaching
        the final waypoint, _reached_goal is set to True.

        If _waypoints is empty, the vehicle moves in its current direction with
        the given _target_speed.

        If _init_speed is True, the control command is post-processed to ensure that
        the initial actor velocity is maintained independent of physics.
        """
        self._reached_goal = False

        if self._waypoints_updated:
            self._waypoints_updated = False
            self._update_plan()

        if self._offset_updated:
            self._offset_updated = False
            self._update_offset()

        target_speed = self._target_speed
        # If target speed is negavite, raise an exception
        if target_speed < 0:
            raise NotImplementedError("Negative target speeds are not yet supported")

        # If distance is less than the proximity threshold, adapt velocity
        if self._obstacle_distance < self._proximity_threshold:
            if "vehicle" in self._obstacle_actor.type_id:
                self._obstacle_distance = float('inf')
                self._obstacle_actor = None
                target_speed = 0

        self._local_planner.set_speed(target_speed * 3.6)
        control = self._local_planner.run_step(debug=False)

        # Check if the actor reached the end of the plan
        if self._local_planner.done():
            self._reached_goal = True

        self._actor.apply_control(control)

        current_speed = math.sqrt(self._actor.get_velocity().x**2 + self._actor.get_velocity().y**2)

        if self._init_speed:

            # If _init_speed is set, and the PID controller is not yet up to the point to take over,
            # we manually set the vehicle to drive with the correct velocity
            if abs(target_speed - current_speed) > 3:
                yaw = self._actor.get_transform().rotation.yaw * (math.pi / 180)
                vx = math.cos(yaw) * target_speed
                vy = math.sin(yaw) * target_speed
                self._actor.set_target_velocity(carla.Vector3D(vx, vy, 0))

        # Change Brake light state
        if (current_speed > target_speed or target_speed < 0.2) and not self._brake_lights_active:
            light_state = self._actor.get_light_state()
            light_state |= carla.VehicleLightState.Brake
            self._actor.set_light_state(carla.VehicleLightState(light_state))
            self._brake_lights_active = True

        if self._brake_lights_active and current_speed < target_speed:
            self._brake_lights_active = False
            light_state = self._actor.get_light_state()
            light_state &= ~carla.VehicleLightState.Brake
            self._actor.set_light_state(carla.VehicleLightState(light_state))
