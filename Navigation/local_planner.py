#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.

from enum import Enum
from collections import deque

import carla

from Navigation.controller import *

class TOPO_OPTIONS(Enum):
    """
    TODO:
    """
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3


class LocalPlanner(object):
    """
    TODO:
    """
    MIN_DISTANCE_PERCENTAGE = 0.9

    def __init__(self, vehicle):
        """

        :param vehicle:
        """
        self._vehicle = vehicle
        self._map = self._vehicle.get_world().get_map()

        self._dt = None
        self._target_speed = None
        self._sampling_radius = None
        self._min_distance = None
        self._current_waypoint = None
        self._next_waypoints = None
        self._vehicle_controller = None
        self._waypoints_queue = deque(maxlen=200)
        self._count = 0

        self.init_controller()

    def __del__(self):
        self._vehicle.destroy()
        print("Calling del!")

    def init_controller(self):
        self._dt = 1.0 / 60.0 #61.0
        self._target_speed = 10.0  # Km/h
        self._sampling_radius = self._target_speed * 0.5 / 3.6
        self._min_distance = self._sampling_radius * self.MIN_DISTANCE_PERCENTAGE

        self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())

        self._vehicle_controller = VehiclePIDController(self._vehicle,
                                                        args_lateral={'K_P': 1.9, 'K_D': 0.0, 'K_I': 0., 'dt': self._dt},
                                                        args_longitudinal={'K_P': 50.0, 'K_D': 0, 'K_I': 1e3, 'dt': self._dt})
        self._vehicle_controller.warmup()

        # compute initial waypoints
        self._waypoints_queue.append(self._current_waypoint.next(self._sampling_radius)[0])
        self._compute_next_waypoints(k=200)


    def check_distance_ith_waypoint(self, point, index):
        if index >= len(self._waypoints_queue):
            return -1.0

        waypoint = self._waypoints_queue[index]
        dx = waypoint.transform.location.x - point.x
        dy = waypoint.transform.location.y - point.y

        return math.sqrt(dx * dx + dy * dy)

    def _compute_next_waypoints(self, k=1):
        # check we do not overflow the queue
        available_entries = self._waypoints_queue.maxlen - len(self._waypoints_queue)
        k = min(available_entries, k)

        for i in range(k):
            last_waypoint = self._waypoints_queue[-1]
            next_waypoints = list(last_waypoint.next(self._sampling_radius))

            if len(next_waypoints) == 1:
                next_waypoint = next_waypoints[0]
            else:
                topo_options = retrieve_options(next_waypoints, last_waypoint)
                # preference to straight
                if TOPO_OPTIONS.STRAIGHT in topo_options:
                    next_waypoint = next_waypoints[topo_options.index(TOPO_OPTIONS.STRAIGHT)]
                elif TOPO_OPTIONS.LEFT in topo_options:
                    next_waypoint = next_waypoints[topo_options.index(TOPO_OPTIONS.LEFT)]
                else:
                    next_waypoint = next_waypoints[topo_options.index(TOPO_OPTIONS.RIGHT)]

            self._waypoints_queue.append(next_waypoint)

    def _resample_waypoints(self, waypoints_queue, current_speed):
        samples = [ waypoints_queue[0] ]
        sample_distance = self._sampling_radius * current_speed
        for elem in waypoints_queue:
            last_sampled = samples[-1]
            if distance(elem, last_sampled) >= sample_distance:
                samples.append(elem)

        return samples

    def run_step(self):
        # not enough waypoints in the horizon? => add more!
        if len(self._waypoints_queue) < int(self._waypoints_queue.maxlen * 0.5):
            self._compute_next_waypoints(k=10)
        #waypoints_resampled = self._resample_waypoints(self._waypoints_queue, get_speed(self._vehicle))

        self._count += 1
        # vehicle waypoint
        self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        # target waypoint
        self._target_waypoint = self._waypoints_queue[0]
        # move using PID controllers
        diff = self._vehicle_controller.run_iter(self._target_speed, self._target_waypoint, self._min_distance*0.80, 4)
        #print("--- Tick! {} --> queue size = {} --> d = {}".format(self._count, len(self._waypoints_queue), diff))

        # purge the queue of obsolete waypoints
        vehicle_transform = self._vehicle.get_transform()
        max_index = -1
        for i, waypoint in enumerate(self._waypoints_queue):
            if distance_vehicle(waypoint, vehicle_transform) < self._min_distance:
                max_index = i

        if max_index >= 0:
            for i in range(max_index + 1):
                self._waypoints_queue.popleft()
                #print('>>> Waypoint poped out!')

        if self._count % 60 == 0:
            #draw_waypoints(self._vehicle.get_world(), [self._target_waypoint], z=40)
            pass#draw_waypoints(self._vehicle.get_world(), self._waypoints_queue, z=40)


def draw_waypoints(world, waypoints, z=0.5):
    for w in waypoints:
        t = w.transform
        begin = t.location + carla.Location(z=z)
        angle = math.radians(t.rotation.yaw)
        end = begin + carla.Location(x=math.cos(angle), y=math.sin(angle))
        world.debug.draw_arrow(begin, end, arrow_size=0.3, life_time=1.0)

def retrieve_options(list_waypoints, current_waypoint):
    options = []
    for next_waypoint in list_waypoints:
        # this is needed because something we are linking to
        # the beggining of an intersection, therefore the
        # variation in angle is small
        #pdb.set_trace()
        next_next_waypoint = next_waypoint.next(3.0)[0]
        link = compute_connection(current_waypoint, next_next_waypoint)
        options.append(link)

    return options

def compute_connection(current_waypoint, next_waypoint):
    n_ = next_waypoint.transform.rotation.yaw
    n_ = n_ % 360.0

    c_ = current_waypoint.transform.rotation.yaw
    c_ = c_ % 360.0

    diff_angle = (n_ - c_) % 180.0
    if diff_angle < 1.0:
        return TOPO_OPTIONS.STRAIGHT
    elif diff_angle > 90.0:
        return TOPO_OPTIONS.LEFT
    else:
        return TOPO_OPTIONS.RIGHT
