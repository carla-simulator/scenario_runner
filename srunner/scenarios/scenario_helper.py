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
from agents.navigation.local_planner import RoadOption

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.atomic_scenario_behavior import *


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


def generate_target_waypoint_list(waypoint, turn=0):
    """
    This method follow waypoints to a junction and choose path based on turn input.
    Turn input: LEFT -> -1, RIGHT -> 1, STRAIGHT -> 0
    @returns a waypoint list from the starting point to the end point according to turn input
    """
    sampling_radius = 1
    reached_junction = False
    wp_list = []
    threshold = math.radians(0.1)
    plan = []
    while True:
        wp_choice = waypoint.next(2)
        if len(wp_choice) > 1:
            reached_junction = True
            waypoint = choose_at_junction(waypoint, wp_choice, turn)
        else:
            waypoint = wp_choice[0]
        plan.append((waypoint, RoadOption.LANEFOLLOW))
        #   End condition for the behavior
        if turn != 0 and reached_junction and len(plan) >= 3:
            v_1 = vector(
                plan[-2][0].transform.location,
                plan[-1][0].transform.location)
            v_2 = vector(
                plan[-3][0].transform.location,
                plan[-2][0].transform.location)
            angle_wp = math.acos(
                np.dot(v_1, v_2) / abs((np.linalg.norm(v_1) * np.linalg.norm(v_2))))
            if angle_wp < threshold:
                break
        elif reached_junction and not plan[-1][0].is_intersection:
            break

    return plan, plan[-1][0]


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
            cos_wp = vec_dots / abs((np.linalg.norm(v_1) * np.linalg.norm(v_2)))
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


def detect_lane_obstacle(actor, extention_factor=3):
    """
    This function identifies if an obstacle is present in front of the reference actor
    """
    world = CarlaDataProvider.get_world()
    world_actors = world.get_actors().filter('vehicle.*')
    actor_bbox = actor.bounding_box
    actor_transform = actor.get_transform()
    actor_location = actor_transform.location
    actor_vector = actor_transform.rotation.get_forward_vector()
    actor_vector = np.array([actor_vector.x, actor_vector.y])
    actor_vector = actor_vector / np.linalg.norm(actor_vector)
    actor_vector = actor_vector*(extention_factor-1)*actor_bbox.extent.x
    actor_location = actor_location + carla.Location(actor_vector[0], actor_vector[1])
    actor_yaw = actor_transform.rotation.yaw

    is_hazard = False
    for adversary in world_actors:
        if adversary.id != actor.id:
            adversary_bbox = adversary.bounding_box
            adversary_transform = adversary.get_transform()
            adversary_loc = adversary_transform.location
            adversary_yaw = adversary_transform.rotation.yaw
            overlap_area = intersection_area(
                (adversary_loc.x, adversary_loc.y,
                 2*adversary_bbox.extent.x, 2*adversary_bbox.extent.y, adversary_yaw),
                (actor_location.x, actor_location.y,
                 2*actor_bbox.extent.x*extention_factor, 2*actor_bbox.extent.y, actor_yaw))
            if  overlap_area > 0:
                is_hazard = True

    return is_hazard


class Vector:
    """
    Simple Vector class
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, v):
        if not isinstance(v, Vector):
            return NotImplemented
        return Vector(self.x + v.x, self.y + v.y)

    def __sub__(self, v):
        if not isinstance(v, Vector):
            return NotImplemented
        return Vector(self.x - v.x, self.y - v.y)

    def cross(self, v):
        if not isinstance(v, Vector):
            return NotImplemented
        return self.x*v.y - self.y*v.x


class Line:
    # ax + by + c = 0
    def __init__(self, v1, v2):
        self.a = v2.y - v1.y
        self.b = v1.x - v2.x
        self.c = v2.cross(v1)

    def __call__(self, p):
        return self.a*p.x + self.b*p.y + self.c

    def intersection(self, other):

        if not isinstance(other, Line):
            return NotImplemented
        w = self.a*other.b - self.b*other.a
        return Vector(
            (self.b*other.c - self.c*other.b)/w,
            (self.c*other.a - self.a*other.c)/w
        )


def rectangle_vertices(cx, cy, w, h, r):
    """
    Converting from (center, dimension, rotation) to vertices' coordinates
    """
    angle = pi*r/180
    dx = w/2
    dy = h/2
    dxcos = dx*cos(angle)
    dxsin = dx*sin(angle)
    dycos = dy*cos(angle)
    dysin = dy*sin(angle)
    return (
        Vector(cx, cy) + Vector(-dxcos - -dysin, -dxsin + -dycos),
        Vector(cx, cy) + Vector(dxcos - -dysin, dxsin + -dycos),
        Vector(cx, cy) + Vector(dxcos -  dysin, dxsin +  dycos),
        Vector(cx, cy) + Vector(-dxcos -  dysin, -dxsin +  dycos)
    )

def intersection_area(r1, r2):
    """
    Overlap area calculation
    """
    # r1 and r2 are in (center, width, height, rotation) representation
    # First convert these into a sequence of vertices
    rect1 = rectangle_vertices(*r1)
    rect2 = rectangle_vertices(*r2)

    # Use the vertices of the first rectangle as
    # starting vertices of the intersection polygon.
    intersection = rect1

    # Loop over the edges of the second rectangle
    for p, q in zip(rect2, rect2[1:] + rect2[:1]):
        if len(intersection) <= 2:
            break # No intersection

        line = Line(p, q)

        # Any point p with line(p) <= 0 is on the "inside" (or on the boundary),
        # any point p with line(p) > 0 is on the "outside".

        # Loop over the edges of the intersection polygon,
        # and determine which part is inside and which is outside.
        new_intersection = []
        line_values = [line(t) for t in intersection]
        for s, t, s_value, t_value in zip(
                intersection, intersection[1:] + intersection[:1],
                line_values, line_values[1:] + line_values[:1]):
            if s_value <= 0:
                new_intersection.append(s)
            if s_value * t_value < 0:
                # Points are on opposite sides.
                # Add the intersection of the lines to new_intersection.
                intersection_point = line.intersection(Line(s, t))
                new_intersection.append(intersection_point)

        intersection = new_intersection

    # Calculate area
    if len(intersection) <= 2:
        return 0

    return 0.5 * sum(p.x*q.y - p.y*q.x for p, q in
                     zip(intersection, intersection[1:] + intersection[:1]))
