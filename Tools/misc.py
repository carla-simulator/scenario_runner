#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" Module with auxiliary functions. """

import math

import numpy as np

import carla


def draw_waypoints(world, waypoints, z=0.5):
    """
    Draw a list of waypoints at a certain height given in z.

    :param world: carla.world object
    :param waypoints: list or iterable container with the waypoints to draw
    :param z: height in meters
    :return:
    """
    for w in waypoints:
        t = w.transform
        begin = t.location + carla.Location(z=z)
        angle = math.radians(t.rotation.yaw)
        end = begin + carla.Location(x=math.cos(angle), y=math.sin(angle))
        world.debug.draw_arrow(begin, end, arrow_size=0.3, life_time=1.0)


def get_speed(vehicle):
    """
    Compute speed of a vehicle in Kmh
    :param vehicle: the vehicle for which speed is calculated
    :return: speed as a float in Kmh
    """
    vel = vehicle.get_velocity()
    return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)


def is_within_distance_ahead(
        target_location, current_location, orientation, max_distance):
    """
    Check if a target object is within a certain distance in front of a reference object.

    :param target_location: location of the target object
    :param current_location: location of the reference object
    :param orientation: orientation of the reference object
    :param max_distance: maximum allowed distance
    :return: True if target object is within max_distance ahead of the reference object
    """
    target_vector = np.array([target_location.x -
                              current_location.x, target_location.y -
                              current_location.y])
    norm_target = np.linalg.norm(target_vector)
    if norm_target > max_distance:
        return False

    forward_vector = np.array(
        [math.cos(math.radians(orientation)), math.sin(math.radians(orientation))])
    d_angle = math.degrees(
        math.acos(
            np.dot(
                forward_vector,
                target_vector) /
            norm_target))

    return d_angle < 90.0


def distance_vehicle(waypoint, vehicle_transform):
    loc = vehicle_transform.location
    dx = waypoint.transform.location.x - loc.x
    dy = waypoint.transform.location.y - loc.y

    return math.sqrt(dx * dx + dy * dy)
