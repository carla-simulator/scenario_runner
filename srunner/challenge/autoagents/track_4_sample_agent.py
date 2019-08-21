#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides an example for a Track4 agent to control the ego vehicle via keyboard
"""


from threading import Thread
import math
import sys
import time

try:
    import pygame
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

import carla


from srunner.challenge.autoagents.autonomous_agent import AutonomousAgent, Track
from srunner.challenge.autoagents.human_agent import KeyboardControl


if sys.version_info >= (3, 3):

    import shutil

    def print_over_same_line(text):
        """
        Refresh text line
        """
        terminal_width = shutil.get_terminal_size((80, 20)).columns
        empty_space = max(0, terminal_width - len(text))
        sys.stdout.write('\r' + text + empty_space * ' ')
        sys.stdout.flush()

else:

    # Workaround for older Python versions.
    def print_over_same_line(text):
        """
        Refresh text line
        """
        line_length = max(print_over_same_line.last_line_length, len(text))
        empty_space = max(0, line_length - len(text))
        sys.stdout.write('\r' + text + empty_space * ' ')
        sys.stdout.flush()
        print_over_same_line.last_line_length = line_length
        print_over_same_line.last_line_length = 0


def distance_vehicle(waypoint, vehicle_position):
    """
    Calculate distance between waypoint and vehicle position
    """
    dx = waypoint[0] - vehicle_position[0]
    dy = waypoint[1] - vehicle_position[1]
    dz = waypoint[2] - vehicle_position[2]

    return math.sqrt(dx * dx + dy * dy + dz * dz)


def get_closest_waypoint(gps_position, scene_layout):
    """
    Get closest waypoint to current gps position
    """
    min_dist = 10000
    closest_way_id = None
    for waypoint_id, waypoint_data in scene_layout.items():
        current_waypoint_distance = distance_vehicle(waypoint_data['position'], gps_position)
        if current_waypoint_distance < min_dist:
            closest_way_id = waypoint_id
            min_dist = current_waypoint_distance

    return closest_way_id, min_dist


class HumanTextInterface(object):

    """
    Class to control a vehicle manually for debugging purposes
    """

    def __init__(self, parent):
        self.quit = False
        self._parent = parent
        self._width = 800
        self._height = 600
        self._throttle_delta = 0.05
        self._steering_delta = 0.01

        pygame.init()
        pygame.font.init()
        self._clock = pygame.time.Clock()
        self._display = pygame.display.set_mode((self._width, self._height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("Track4 Sample Agent")

    def run(self):
        """
        Run the GUI
        """
        while not self._parent.agent_engaged:
            time.sleep(0.5)

        controller = KeyboardControl()

        input_data = self._parent.sensor_interface.get_data()
        # agent is engaged. Take the closest waypoint.

        closest_waypoint, distance = get_closest_waypoint(input_data['GPS'][1],
                                                          input_data['scene_layout'][1])

        # We navigate now iterating from this
        while not self.quit:

            self._clock.tick_busy_loop(20)
            controller.parse_events(self._parent.current_control, self._clock)
            # Process events
            pygame.event.pump()  # to get all the keyboard control
            # process sensor data
            input_data = self._parent.sensor_interface.get_data()

            # merge your position with the input data and inform the client
            # Your position
            print("Closest waypoint id is ", closest_waypoint, ' Dist ', distance)

        pygame.quit()


class Track4SampleAgent(AutonomousAgent):

    """
    THis is a human controlled agent with track 4 access for testing
    """

    current_control = None
    agent_engaged = False

    def setup(self, path_to_conf_file):
        """
        Setup the agent parameters
        """
        self.track = Track.SCENE_LAYOUT
        self.agent_engaged = False
        self.current_control = carla.VehicleControl()
        self.current_control.steer = 0.0
        self.current_control.throttle = 1.0
        self.current_control.brake = 0.0
        self.current_control.hand_brake = False
        self._hic = HumanTextInterface(self)
        self._thread = Thread(target=self._hic.run)
        self._thread.start()

    def sensors(self):
        """
        Define the sensor suite required by the agent
        :return: a list containing the required sensors in the following format:

        """
        sensors = [
            {'type': 'sensor.other.gnss', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'id': 'GPS'},
                   {'type': 'sensor.scene_layout', 'id': 'scene_layout'},
                   {'type': 'sensor.object_finder', 'reading_frequency': 20, 'id': 'object_finder'},
        ]

        return sensors

    def run_step(self, input_data, timestamp):
        """
        Execute one step of navigation.
        """
        self.agent_engaged = True
        return self.current_control

    def destroy(self):
        """
        Cleanup
        """
        self._hic.quit = True
        self._thread.join()
