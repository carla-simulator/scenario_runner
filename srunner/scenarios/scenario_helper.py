#!/usr/bin/env python

# Copyright (c) 2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Summary of useful helper functions for scenarios
"""

from __future__ import print_function
import math

import numpy as np
import carla
from agents.tools.misc import vector, is_within_distance_ahead

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider


def get_crossing_point(actor):
    """
    Get the next crossing point location in front of the ego vehicle

    @return point of crossing
    """
    wp_cross = CarlaDataProvider.get_map().get_waypoint(actor.get_location())

    while(not wp_cross.is_intersection):
        wp_cross = wp_cross.next(2)[0]

    crossing = carla.Location(x=wp_cross.transform.location.x,
                              y=wp_cross.transform.location.y, z=wp_cross.transform.location.z)

    return crossing


def get_geometric_linear_intersection(ego_actor, other_actor):
    """
    Obtain a intersection point between two actor's location by using their waypoints (wp)

    @return point of intersection of the two vehicles
    """

    wp_ego_1 = CarlaDataProvider.get_map().get_waypoint(ego_actor.get_location())
    wp_ego_2 = wp_ego_1.next(1)[0]
    x_ego_1 = wp_ego_1.transform.location.x
    y_ego_1 = wp_ego_1.transform.location.y
    x_ego_2 = wp_ego_2.transform.location.x
    y_ego_2 = wp_ego_2.transform.location.y

    wp_other_1 = CarlaDataProvider.get_world().get_map().get_waypoint(other_actor.get_location())
    wp_other_2 = wp_other_1.next(1)[0]
    x_other_1 = wp_other_1.transform.location.x
    y_other_1 = wp_other_1.transform.location.y
    x_other_2 = wp_other_2.transform.location.x
    y_other_2 = wp_other_2.transform.location.y

    s = np.vstack([(x_ego_1, y_ego_1), (x_ego_2, y_ego_2), (x_other_1, y_other_1), (x_other_2, y_other_2)])
    h = np.hstack((s, np.ones((4, 1))))
    line1 = np.cross(h[0], h[1])
    line2 = np.cross(h[2], h[3])
    x, y, z = np.cross(line1, line2)
    if z == 0:
        return (float('inf'), float('inf'))

    intersection = carla.Location(x=x / z, y=y / z, z=0)

    return intersection


def get_location_in_distance(actor, distance):
    """
    Obtain a location in a given distance from the current actor's location.
    Note: Search is stopped on first intersection.

    @return obtained location and the traveled distance
    """
    waypoint = CarlaDataProvider.get_map().get_waypoint(actor.get_location())
    traveled_distance = 0
    while not waypoint.is_intersection and traveled_distance < distance:
        waypoint_new = waypoint.next(1.0)[-1]
        traveled_distance += waypoint_new.transform.location.distance(waypoint.transform.location)
        waypoint = waypoint_new

    return waypoint.transform.location, traveled_distance


def get_location_in_distance_from_wp(waypoint, distance):
    """
    Obtain a location in a given distance from the current actor's location.
    Note: Search is stopped on first intersection.

    @return obtained location and the traveled distance
    """
    traveled_distance = 0
    while not waypoint.is_intersection and traveled_distance < distance:
        waypoint_new = waypoint.next(1.0)[-1]
        traveled_distance += waypoint_new.transform.location.distance(waypoint.transform.location)
        waypoint = waypoint_new

    return waypoint.transform.location, traveled_distance


def get_waypoint_in_distance(waypoint, distance):
    """
    Obtain a waypoint in a given distance from the current actor's location.
    Note: Search is stopped on first intersection.
    @return obtained waypoint and the traveled distance
    """
    traveled_distance = 0
    while not waypoint.is_intersection and traveled_distance < distance:
        waypoint_new = waypoint.next(1.0)[-1]
        traveled_distance += waypoint_new.transform.location.distance(waypoint.transform.location)
        waypoint = waypoint_new

    return waypoint, traveled_distance


def generate_target_waypoint(waypoint, turn=0):
    """
    This method follow waypoints to a junction and choose path based on turn input.
    Turn input: LEFT -> -1, RIGHT -> 1, STRAIGHT -> 0
    @returns a waypoint list according to turn input
    """
    sampling_radius = 1
    reached_junction = False
    wp_list = []
    threshold = math.radians(0.1)
    while True:

        wp_choice = waypoint.next(sampling_radius)
        #   Choose path at intersection
        if len(wp_choice) > 1:
            reached_junction = True
            waypoint = choose_at_junction(waypoint, wp_choice, turn)
        else:
            waypoint = wp_choice[0]
        wp_list.append(waypoint)
        #   End condition for the behavior
        if turn != 0 and reached_junction and len(wp_list) >= 3:
            v_1 = vector(
                wp_list[-2].transform.location,
                wp_list[-1].transform.location)
            v_2 = vector(
                wp_list[-3].transform.location,
                wp_list[-2].transform.location)
            vec_dots = np.dot(v_1, v_2)
            cos_wp = vec_dots/abs((np.linalg.norm(v_1)*np.linalg.norm(v_2)))
            angle_wp = math.acos(min(1.0, cos_wp))  # COS can't be larger than 1, it can happen due to float imprecision
            if angle_wp < threshold:
                break
        elif reached_junction and not wp_list[-1].is_intersection:
            break
    return wp_list[-1]


def choose_at_junction(current_waypoint, next_choices, direction=0):
    """
    This function chooses the appropriate waypoint from next_choices based on direction
    """
    current_transform = current_waypoint.transform
    current_location = current_transform.location
    projected_location = current_location + \
        carla.Location(
            x=math.cos(math.radians(current_transform.rotation.yaw)),
            y=math.sin(math.radians(current_transform.rotation.yaw)))
    current_vector = vector(current_location, projected_location)
    cross_list = []
    cross_to_waypoint = dict()
    for waypoint in next_choices:
        waypoint = waypoint.next(10)[0]
        select_vector = vector(current_location, waypoint.transform.location)
        cross = np.cross(current_vector, select_vector)[2]
        cross_list.append(cross)
        cross_to_waypoint[cross] = waypoint
    select_cross = None
    if direction > 0:
        select_cross = max(cross_list)
    elif direction < 0:
        select_cross = min(cross_list)
    else:
        select_cross = min(cross_list, key=abs)

    return cross_to_waypoint[select_cross]


def get_intersection(ego_actor, other_actor):
    """
    Obtain a intersection point between two actor's location
    @return the intersection location
    """
    waypoint = CarlaDataProvider.get_map().get_waypoint(ego_actor.get_location())
    waypoint_other = CarlaDataProvider.get_map().get_waypoint(other_actor.get_location())
    max_dist = float("inf")
    distance = float("inf")
    while distance <= max_dist:
        max_dist = distance
        current_location = waypoint.transform.location
        waypoint_choice = waypoint.next(1)
        #   Select the straighter path at intersection
        if len(waypoint_choice) > 1:
            max_dot = -1 * float('inf')
            loc_projection = current_location + carla.Location(
                x=math.cos(math.radians(waypoint.transform.rotation.yaw)),
                y=math.sin(math.radians(waypoint.transform.rotation.yaw)))
            v_current = vector(current_location, loc_projection)
            for wp_select in waypoint_choice:
                v_select = vector(current_location, wp_select.transform.location)
                dot_select = np.dot(v_current, v_select)
                if dot_select > max_dot:
                    max_dot = dot_select
                    waypoint = wp_select
        else:
            waypoint = waypoint_choice[0]
        distance = current_location.distance(waypoint_other.transform.location)

    return current_location


def detect_lane_obstacle(actor, max_distance=10):
    """
    This function identifies if an obstacle is present in front of the reference actor
    """
    world = CarlaDataProvider.get_world()
    wmap = CarlaDataProvider.get_map()
    world_actors = world.get_actors().filter('vehicle.*')
    reference_vehicle_location = actor.get_location()
    reference_vehicle_waypoint = wmap.get_waypoint(reference_vehicle_location)
    is_hazard = False
    for adversary in world_actors:
        if adversary.id != actor.id:
            # if the object is not in our lane it's not an obstacle
            waypoint = wmap.get_waypoint(adversary.get_location())
            if waypoint.road_id == reference_vehicle_waypoint.road_id and \
                    waypoint.lane_id == reference_vehicle_waypoint.lane_id and\
                        is_within_distance_ahead(
                                waypoint.transform.location,
                                reference_vehicle_waypoint.transform.location,
                                reference_vehicle_waypoint.transform.rotation.yaw,
                                max_distance):
                is_hazard = True
                break

    return is_hazard
