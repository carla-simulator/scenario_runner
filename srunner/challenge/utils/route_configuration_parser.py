from __future__ import print_function
import math
import json
import xml.etree.ElementTree as ET
import carla
"""
    Module use to parse all the route and scenario configuration parameters .
"""

# TODO  check this threshold, it could be a bit larger but not so large that we cluster scenarios.
TRIGGER_THRESHOLD = 5.0   # Threshold to say if a trigger position is new or repeated, works for matching positions
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
        distance = math.sqrt(dx*dx + dy*dy)
        dyaw = trigger['yaw'] - trigger['yaw']
        dist_angle = math.sqrt(dyaw * dyaw)
        if distance < TRIGGER_THRESHOLD and dist_angle < TRIGGER_ANGLE_THRESHOLD:
            return trigger_id

    return None


def convert_waypoint_float(waypoint):

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
        def match_waypoints(w1, wtransform):
            dx = float(w1['x']) - wtransform.location.x
            dy = float(w1['y']) - wtransform.location.y
            dz = float(w1['z']) - wtransform.location.z
            dist_position = math.sqrt(dx * dx + dy * dy + dz * dz)

            dyaw = float(w1['yaw']) - wtransform.rotation.yaw

            dist_angle = math.sqrt(dyaw * dyaw)

            return dist_position < TRIGGER_THRESHOLD and dist_angle < TRIGGER_ANGLE_THRESHOLD


        # TODO this function can be optimized to run on Log(N) time
        for route_waypoint in route_description:
            if match_waypoints(world_location, route_waypoint[0]):
                return True

        return False


def match_scenario_route(scenario_type, other_actors, trigger_point, trajectory):


    # All the scenarios that are not on intersection always affect the route
    if scenario_type == 'Scenario1' or scenario_type == 'Scenario2' or scenario_type == 'Scenario3' \
            or scenario_type == 'Scenario5' or scenario_type == 'Scenario6':

        return True

    elif scenario_type == 'Scenario4':
        return True

    else:
        return True




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
            scenario_type = scenario["scenario_type"]
            for event in scenario["available_event_configurations"]:
                waypoint = event['transform'] # trigger point of this scenario
                convert_waypoint_float(waypoint)
                if match_world_location_to_route(waypoint, route_description['trajectory']):
                    # We match trigger point to the  route, now we need to check if the route affects
                    if match_scenario_route(scenario["scenario_type"], event['other_actors'],
                                            waypoint, route_description['trajectory']):
                        # We match a location for this scenario, create a scenario object so this scenario
                        # can be instantiated later

                        if 'other_actors' in event:
                            other_vehicles = event['other_actors']
                        else:
                            other_vehicles = None

                        scenario_description = {
                                               'name': scenario_type,
                                               'other_actors': other_vehicles,
                                               'trigger_position': waypoint
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

