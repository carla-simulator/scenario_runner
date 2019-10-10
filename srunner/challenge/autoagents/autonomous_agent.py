#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides the base class for all autonomous agents
"""

from __future__ import print_function

from enum import Enum

import carla

from srunner.challenge.envs.sensor_interface import SensorInterface
from srunner.scenariomanager.timer import GameTime
from srunner.challenge.utils.route_manipulation import downsample_route


class Track(Enum):

    """
    This enum represents the different tracks of the CARLA AD challenge.
    """
    ALL_SENSORS = 1
    CAMERAS = 2
    ALL_SENSORS_HDMAP_WAYPOINTS = 3
    SCENE_LAYOUT = 4


class AutonomousAgent(object):

    """
    Autonomous agent base class. All user agents have to be derived from this class
    """

    def __init__(self, path_to_conf_file):
        self.track = Track.CAMERAS
        #  current global plans to reach a destination
        self._global_plan = None
        self._global_plan_world_coord = None

        # this data structure will contain all sensor data
        self.sensor_interface = SensorInterface()

        # agent's initialization
        self.setup(path_to_conf_file)

    def setup(self, path_to_conf_file):
        """
        Initialize everything needed by your agent and set the track attribute to the right type:
            Track.ALL_SENSORS : LIDAR, cameras, GPS and speed sensor allowed
            Track.CAMERAS : Only cameras and GPS allowed
            Track.ALL_SENSORS_HDMAP_WAYPOINTS : All sensors and HD Map and waypoints allowed
            Track.SCENE_LAYOUT : No sensors allowed, the agent receives a high-level representation of the scene.
        """
        pass

    def sensors(self):  # pylint: disable=no-self-use
        """
        Define the sensor suite required by the agent

        :return: a list containing the required sensors in the following format:

        [
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Left'},

            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Right'},

            {'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
             'id': 'LIDAR'}
        ]

        """
        sensors = []

        return sensors

    def run_step(self, input_data, timestamp):
        """
        Execute one step of navigation.
        :return: control
        """
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 0.0
        control.hand_brake = False

        return control

    def destroy(self):
        """
        Destroy (clean-up) the agent
        :return:
        """
        pass

    def __call__(self):
        """
        Execute the agent call, e.g. agent()
        Returns the next vehicle controls
        """
        input_data = self.sensor_interface.get_data()

        timestamp = GameTime.get_time()
        wallclock = GameTime.get_wallclocktime()
        print('======[Agent] Wallclock_time = {} / Sim_time = {}'.format(wallclock, timestamp))

        control = self.run_step(input_data, timestamp)
        control.manual_gear_shift = False

        return control

    def all_sensors_ready(self):
        """
        Check if all sensors are ready
        Returns true if sensors are ready
        """
        return self.sensor_interface.all_sensors_ready()

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        """
        Set the plan (route) for the agent
        """

        if self.track == Track.CAMERAS or self.track == Track.ALL_SENSORS:
            ds_ids = downsample_route(global_plan_world_coord, 32)

            self._global_plan_world_coord = [(global_plan_world_coord[x][0], global_plan_world_coord[x][1])
                                             for x in ds_ids]
            self._global_plan = [global_plan_gps[x] for x in ds_ids]

        else:   # No downsampling is performed

            self._global_plan = global_plan_gps
            self._global_plan_world_coord = global_plan_world_coord
