#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Traffic light helpers: where a light's trigger volume sits, how the lights of a
junction relate to one another, and how to drive and restore their states.
"""

import math

import carla

from srunner.scenariomanager.simulation_data import SimulationData


def get_trafficlight_trigger_location(traffic_light):  # pylint: disable=invalid-name
    """
    Calculates the yaw of the waypoint that represents the trigger volume of the traffic light
    """

    def rotate_point(point, angle):
        """
        rotate a given point by a given angle
        """
        x_ = math.cos(math.radians(angle)) * point.x - math.sin(math.radians(angle)) * point.y
        y_ = math.sin(math.radians(angle)) * point.x - math.cos(math.radians(angle)) * point.y

        return carla.Vector3D(x_, y_, point.z)

    base_transform = traffic_light.get_transform()
    base_rot = base_transform.rotation.yaw
    area_loc = base_transform.transform(traffic_light.trigger_volume.location)
    area_ext = traffic_light.trigger_volume.extent

    point = rotate_point(carla.Vector3D(0, 0, area_ext.z), base_rot)
    point_location = area_loc + carla.Location(x=point.x, y=point.y)

    return carla.Location(point_location.x, point_location.y, point_location.z)


def annotate_trafficlight_in_group(traffic_light):
    # type: (carla.TrafficLight) -> dict[str, list[carla.TrafficLight]]
    """
    Get dictionary with traffic light group info for a given traffic light
    """
    dict_annotations = {'ref': [], 'opposite': [], 'left': [], 'right': []}

    # Get the waypoints
    ref_location = get_trafficlight_trigger_location(traffic_light)
    ref_waypoint = SimulationData.get_map().get_waypoint(ref_location)
    ref_yaw = ref_waypoint.transform.rotation.yaw

    group_tl = traffic_light.get_group_traffic_lights()

    for target_tl in group_tl:
        if traffic_light.id == target_tl.id:
            dict_annotations['ref'].append(target_tl)
        else:
            # Get the angle between yaws
            target_location = get_trafficlight_trigger_location(target_tl)
            target_waypoint = SimulationData.get_map().get_waypoint(target_location)
            target_yaw = target_waypoint.transform.rotation.yaw

            diff = (target_yaw - ref_yaw) % 360

            if diff > 330:
                continue
            elif diff > 225:
                dict_annotations['right'].append(target_tl)
            elif diff > 135.0:
                dict_annotations['opposite'].append(target_tl)
            elif diff > 30:
                dict_annotations['left'].append(target_tl)

    return dict_annotations


def update_light_states(ego_light, annotations, states, freeze=False, timeout=1000000000):
    # type: (carla.TrafficLight, dict[str, list[carla.TrafficLight]], dict[str, carla.TrafficLightState], bool, float) -> list[dict] # pylint: disable=line-too-long
    """
    Update traffic light states
    """
    reset_params = []  # type: list[dict]

    for state in states:
        relevant_lights = []
        if state == 'ego':
            relevant_lights = [ego_light]
        else:
            relevant_lights = annotations[state]
        for light in relevant_lights:
            prev_state = light.get_state()
            prev_green_time = light.get_green_time()
            prev_red_time = light.get_red_time()
            prev_yellow_time = light.get_yellow_time()
            reset_params.append(
                {
                    'light': light,
                    'state': prev_state,
                    'green_time': prev_green_time,
                    'red_time': prev_red_time,
                    'yellow_time': prev_yellow_time,
                }
            )

            light.set_state(states[state])
            if freeze:
                light.set_green_time(timeout)
                light.set_red_time(timeout)
                light.set_yellow_time(timeout)

    return reset_params


def reset_lights(reset_params):
    """
    Reset traffic lights
    """
    for param in reset_params:
        param['light'].set_state(param['state'])
        param['light'].set_green_time(param['green_time'])
        param['light'].set_red_time(param['red_time'])
        param['light'].set_yellow_time(param['yellow_time'])


def get_next_traffic_light(actor):
    """
    Returns the next relevant traffic light for the provided actor
    """
    world = SimulationData.get_world()
    location = actor.get_transform().location

    waypoint = SimulationData.get_map().get_waypoint(location)
    # Create list of all waypoints until next intersection
    list_of_waypoints = []
    while waypoint and not waypoint.is_junction:
        list_of_waypoints.append(waypoint)
        waypoint = waypoint.next(2.0)[0]

    # If the list is empty, the actor is in an intersection
    if not list_of_waypoints:
        return None

    relevant_traffic_light = None
    distance_to_relevant_traffic_light = float("inf")

    for traffic_light in world.get_actors().filter('*traffic_light*'):
        if not hasattr(traffic_light, 'trigger_volume'):
            continue
        tl_t = traffic_light.get_transform()
        transformed_tv = tl_t.transform(traffic_light.trigger_volume.location)
        distance = carla.Location(transformed_tv).distance(list_of_waypoints[-1].transform.location)

        if distance < distance_to_relevant_traffic_light:
            relevant_traffic_light = traffic_light
            distance_to_relevant_traffic_light = distance

    return relevant_traffic_light
