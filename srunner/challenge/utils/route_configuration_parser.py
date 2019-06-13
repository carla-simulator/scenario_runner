#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Module use to parse all the route and scenario configuration parameters .
"""

from __future__ import print_function
import json
import math
import xml.etree.ElementTree as ET

import carla
from agents.navigation.local_planner import RoadOption

# TODO  check this threshold, it could be a bit larger but not so large that we cluster scenarios.
TRIGGER_THRESHOLD = 2.0  # Threshold to say if a trigger position is new or repeated, works for matching positions
TRIGGER_ANGLE_THRESHOLD = 10  # Threshold to say if two angles can be considering matching when matching transforms.


def parse_annotations_file(annotation_filename):
    """
    Return the annotations of which positions where the scenarios are going to happen.
    :param annotation_filename: the filename for the anotations file
    :return:
    """

    with open(annotation_filename, 'r') as f:
        annotation_dict = json.loads(f.read())

    final_dict = {}

    for town_dict in annotation_dict['available_scenarios']:
        final_dict.update(town_dict)

    return final_dict  # the file has a current maps name that is an one element vec


def parse_routes_file(route_filename):
    """
    Returns a list of route elements that is where the challenge is going to happen.
    :param route_filename: the path to a set of routes.
    :return:  List of dicts containing the waypoints, id and town of the routes
    """

    list_route_descriptions = []
    tree = ET.parse(route_filename)
    for route in tree.iter("route"):
        route_town = route.attrib['map']
        route_id = route.attrib['id']
        waypoint_list = []  # the list of waypoints that can be found on this route
        for waypoint in route.iter('waypoint'):
            waypoint_list.append(carla.Location(x=float(waypoint.attrib['x']),
                                                y=float(waypoint.attrib['y']),
                                                z=float(waypoint.attrib['z'])))

            # Waypoints is basically a list of XML nodes

        list_route_descriptions.append({
            'id': route_id,
                                    'town_name': route_town,
                                    'trajectory': waypoint_list
        })

    return list_route_descriptions


def check_trigger_position(new_trigger, existing_triggers):
    """
    Check if this trigger position already exists or if it is a new one.
    :param new_trigger:
    :param existing_triggers:
    :return:
    """

    for trigger_id in existing_triggers.keys():
        trigger = existing_triggers[trigger_id]
        dx = trigger['x'] - new_trigger['x']
        dy = trigger['y'] - new_trigger['y']
        distance = math.sqrt(dx * dx + dy * dy)
        dyaw = trigger['yaw'] - trigger['yaw']
        dist_angle = math.sqrt(dyaw * dyaw)
        if distance < (TRIGGER_THRESHOLD * 2) and dist_angle < TRIGGER_ANGLE_THRESHOLD:
            return trigger_id

    return None


def convert_waypoint_float(waypoint):
    """
    Convert waypoint values to float
    """
    waypoint['x'] = float(waypoint['x'])
    waypoint['y'] = float(waypoint['y'])
    waypoint['z'] = float(waypoint['z'])
    waypoint['yaw'] = float(waypoint['yaw'])


def match_world_location_to_route(world_location, route_description):
    """
    We match this location to a given route.
        world_location:
        route_description:
    """
    def match_waypoints(waypoint1, wtransform):
        """
        Check if waypoint1 and wtransform are similar
        """
        dx = float(waypoint1['x']) - wtransform.location.x
        dy = float(waypoint1['y']) - wtransform.location.y
        dz = float(waypoint1['z']) - wtransform.location.z
        dist_position = math.sqrt(dx * dx + dy * dy + dz * dz)

        dyaw = float(waypoint1['yaw']) - wtransform.rotation.yaw

        dist_angle = math.sqrt(dyaw * dyaw)

        return dist_position < TRIGGER_THRESHOLD and dist_angle < TRIGGER_ANGLE_THRESHOLD

    match_position = 0
    # TODO this function can be optimized to run on Log(N) time
    for route_waypoint in route_description:
        if match_waypoints(world_location, route_waypoint[0]):
            return match_position
        match_position += 1

    return None


def get_scenario_type(scenario, match_position, trajectory):
    """
    Some scenarios have different types depending on the route.
    :param scenario: the scenario name
    :param match_position: the matching position for the scenarion
    :param trajectory: the route trajectory the ego is following
    :return: 0 for option, 0 ,1 for option
    """

    if scenario == 'Scenario4':
        for tuple_wp_turn in trajectory[match_position:]:
            if RoadOption.LANEFOLLOW != tuple_wp_turn[1]:
                if RoadOption.LEFT == tuple_wp_turn[1]:
                    return 1
                elif RoadOption.RIGHT == tuple_wp_turn[1]:
                    return 0
                return None
        return None

    return 0


def scan_route_for_scenarios(route_description, world_annotations):
    """
    Just returns a plain list of possible scenarios that can happen in this route by matching
    the locations from the scenario into the route description

    :return:  A list of scenario definitions with their correspondent parameters
    """

    # the triggers dictionaries:
    existent_triggers = {}
    # We have a table of IDs and trigger positions associated
    possible_scenarios = {}

    # Keep track of the trigger ids being added
    latest_trigger_id = 0

    for town_name in world_annotations.keys():
        if town_name != route_description['town_name']:
            continue

        scenarios = world_annotations[town_name]
        for scenario in scenarios:  # For each existent scenario
            scenario_name = scenario["scenario_type"]
            for event in scenario["available_event_configurations"]:
                waypoint = event['transform']  # trigger point of this scenario
                convert_waypoint_float(waypoint)
                # We match trigger point to the  route, now we need to check if the route affects
                match_position = match_world_location_to_route(waypoint, route_description['trajectory'])
                if match_position is not None:
                    # We match a location for this scenario, create a scenario object so this scenario
                    # can be instantiated later

                    if 'other_actors' in event:
                        other_vehicles = event['other_actors']
                    else:
                        other_vehicles = None
                    scenario_subtype = get_scenario_type(scenario_name, match_position,
                                                         route_description['trajectory'])
                    if scenario_subtype is None:
                        continue
                    scenario_description = {
                        'name': scenario_name,
                                           'other_actors': other_vehicles,
                                           'trigger_position': waypoint,
                                           'type': scenario_subtype,  # some scenarios have different configurations
                    }

                    trigger_id = check_trigger_position(waypoint, existent_triggers)
                    if trigger_id is None:
                        # This trigger does not exist create a new reference on existent triggers
                        existent_triggers.update({latest_trigger_id: waypoint})
                        # Update a reference for this trigger on the possible scenarios
                        possible_scenarios.update({latest_trigger_id: []})
                        trigger_id = latest_trigger_id
                        # Increment the latest trigger
                        latest_trigger_id += 1

                    possible_scenarios[trigger_id].append(scenario_description)

    return possible_scenarios, existent_triggers
