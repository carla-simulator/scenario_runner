#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Module used to parse all the route and scenario configuration parameters.
"""

import json
import math
import xml.etree.ElementTree as ET

import carla
from agents.navigation.local_planner import RoadOption
from srunner.scenarioconfigs.route_scenario_configuration import RouteScenarioConfiguration
from srunner.scenarioconfigs.scenario_configuration import ScenarioConfiguration, ActorConfigurationData

# TODO  check this threshold, it could be a bit larger but not so large that we cluster scenarios.
TRIGGER_THRESHOLD = 2.0  # Threshold to say if a trigger position is new or repeated, works for matching positions
TRIGGER_ANGLE_THRESHOLD = 10  # Threshold to say if two angles can be considering matching when matching transforms.


def convert_dict_to_transform(scenario_dict):
    """Convert a JSON dict to a CARLA transform"""
    return carla.Transform(
        carla.Location(float(scenario_dict['x']), float(scenario_dict['y']), float(scenario_dict['z'])),
        carla.Rotation(roll=0.0, pitch=0.0, yaw=float(scenario_dict['yaw']))
    )


class RouteParser(object):

    """
    Pure static class used to parse all the route and scenario configuration parameters.
    """

    @staticmethod
    def parse_scenario_file_to_dict(scenario_file):
        """
        Parses and returns the scenario file into a dictionary
        :param scenario_file: the filename for the scenario file
        :return:
        """

        with open(scenario_file, 'r', encoding='utf-8') as f:
            scenario_dict = json.loads(f.read())

        final_dict = {}

        for town_dict in scenario_dict['available_scenarios']:
            final_dict.update(town_dict)

        return final_dict  # the file has a current maps name that is an one element vec

    @staticmethod
    def parse_routes_file(route_filename, scenario_file, single_route=None):
        """
        Returns a list of route configuration elements.
        :param route_filename: the path to a set of routes.
        :param single_route: If set, only this route shall be returned
        :return: List of dicts containing the waypoints, id and town of the routes
        """

        list_route_descriptions = []
        tree = ET.parse(route_filename)
        for route in tree.iter("route"):

            route_id = route.attrib['id']
            if single_route and route_id != single_route:
                continue

            new_config = RouteScenarioConfiguration()
            new_config.town = route.attrib['town']
            new_config.name = "RouteScenario_{}".format(route_id)
            new_config.weather = RouteParser.parse_weather(route)
            new_config.scenario_file = scenario_file

            waypoint_list = []  # the list of waypoints that can be found on this route
            for waypoint in route.iter('waypoint'):
                waypoint_list.append(carla.Location(x=float(waypoint.attrib['x']),
                                                    y=float(waypoint.attrib['y']),
                                                    z=float(waypoint.attrib['z'])))

            new_config.trajectory = waypoint_list

            list_route_descriptions.append(new_config)

        return list_route_descriptions

    @staticmethod
    def parse_weather(route):
        """
        Returns a carla.WeatherParameters with the corresponding weather for that route. If the route
        has no weather attribute, the default one is triggered.
        """

        route_weather = route.find("weather")
        if route_weather is None:

            weather = carla.WeatherParameters(sun_altitude_angle=70)

        else:
            weather = carla.WeatherParameters()
            for weather_attrib in route.iter("weather"):

                if 'cloudiness' in weather_attrib.attrib:
                    weather.cloudiness = float(weather_attrib.attrib['cloudiness'])
                if 'precipitation' in weather_attrib.attrib:
                    weather.precipitation = float(weather_attrib.attrib['precipitation'])
                if 'precipitation_deposits' in weather_attrib.attrib:
                    weather.precipitation_deposits = float(weather_attrib.attrib['precipitation_deposits'])
                if 'wind_intensity' in weather_attrib.attrib:
                    weather.wind_intensity = float(weather_attrib.attrib['wind_intensity'])
                if 'sun_azimuth_angle' in weather_attrib.attrib:
                    weather.sun_azimuth_angle = float(weather_attrib.attrib['sun_azimuth_angle'])
                if 'sun_altitude_angle' in weather_attrib.attrib:
                    weather.sun_altitude_angle = float(weather_attrib.attrib['sun_altitude_angle'])
                if 'wetness' in weather_attrib.attrib:
                    weather.wetness = float(weather_attrib.attrib['wetness'])
                if 'fog_distance' in weather_attrib.attrib:
                    weather.fog_distance = float(weather_attrib.attrib['fog_distance'])
                if 'fog_density' in weather_attrib.attrib:
                    weather.fog_density = float(weather_attrib.attrib['fog_density'])
                if 'scattering_intensity' in weather_attrib.attrib:
                    weather.scattering_intensity = float(weather_attrib.attrib['scattering_intensity'])
                if 'mie_scattering_scale' in weather_attrib.attrib:
                    weather.mie_scattering_scale = float(weather_attrib.attrib['mie_scattering_scale'])
                if 'rayleigh_scattering_scale' in weather_attrib.attrib:
                    weather.rayleigh_scattering_scale = float(weather_attrib.attrib['rayleigh_scattering_scale'])

        return weather

    @staticmethod
    def get_trigger_position(scenario_trigger, existing_triggers):
        """
        Check if this trigger position already exists or if it is a new one.
        :param scenario_trigger: position to be checked
        :param existing_triggers: list with all the already found position
        :return:
        """
        for trigger in existing_triggers:
            dx = trigger.location.x - scenario_trigger.location.x
            dy = trigger.location.y - scenario_trigger.location.y
            distance = math.sqrt(dx * dx + dy * dy)

            dyaw = (trigger.rotation.yaw - scenario_trigger.rotation.yaw) % 360
            if distance < TRIGGER_THRESHOLD \
                    and (dyaw < TRIGGER_ANGLE_THRESHOLD or dyaw > (360 - TRIGGER_ANGLE_THRESHOLD)):
                return trigger

        return None

    @staticmethod
    def match_trigger_to_route(trigger_transform, route):
        """
        Check if the scenario is affecting the route.
        This is true if the trigger position is very close to any route point
        """
        def is_trigger_close(trigger_transform, route_transform):
            """Check if the two transforms are similar"""
            dx = trigger_transform.location.x - route_transform.location.x
            dy = trigger_transform.location.y - route_transform.location.y
            dz = trigger_transform.location.z - route_transform.location.z
            dpos = math.sqrt(dx * dx + dy * dy + dz * dz)

            dyaw = (float(trigger_transform.rotation.yaw) - route_transform.rotation.yaw) % 360

            return dpos < TRIGGER_THRESHOLD \
                and (dyaw < TRIGGER_ANGLE_THRESHOLD or dyaw > (360 - TRIGGER_ANGLE_THRESHOLD))

        for position, [route_transform, _] in enumerate(route):
            if is_trigger_close(trigger_transform, route_transform):
                return position

        return None

    @staticmethod
    def get_scenario_subtype(scenario, route):
        """
        Some scenarios have subtypes depending on the route trajectory,
        even being invalid if there isn't a valid one. As an example,
        some scenarios need the route to turn in a specific direction,
        and if this isn't the case, the scenario should not be considered valid.
        This is currently only used for validity purposes.

        :param scenario: the scenario name
        :param route: route starting at the triggering point of the scenario
        :return: tag representing this subtype
        """

        def is_junction_option(option):
            """Whether or not an option is part of a junction"""
            if option in (RoadOption.LANEFOLLOW, RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT):
                return False
            return True

        subtype = None

        if scenario == 'Scenario4':  # Only if the route turns
            for _, option in route:
                if is_junction_option(option):
                    if option == RoadOption.LEFT:
                        subtype = 'S4left'
                    elif option == RoadOption.RIGHT:
                        subtype = 'S4right'
                    else:
                        subtype = None
                    break  # Avoid checking all of them
                subtype = None

        if scenario == 'Scenario7':
            for _, option in route:
                if is_junction_option(option):
                    if RoadOption.STRAIGHT == option:
                        subtype = 'S7opposite'
                    break
        elif scenario == 'Scenario8':  # Only if the route turns left
            for _, option in route:
                if is_junction_option(option):
                    if option == RoadOption.LEFT:
                        subtype = 'S8left'
                    break
        elif scenario == 'Scenario9':  # Only if the route turns right
            for _, option in route:
                if is_junction_option(option):
                    if option == RoadOption.RIGHT:
                        subtype = 'S9right'
                    break
        else:
            subtype = 'valid'

        return subtype

    @staticmethod
    def scan_route_for_scenarios(route_name, trajectory, world_annotations):
        """
        Filters all the scenarios that are affecting the route.
        Returns a dictionary where each item is a list of all the scenarios that are close to each other
        """
        possible_scenarios = {}

        for town_name in world_annotations.keys():
            if town_name != route_name:
                continue

            town_scenarios = world_annotations[town_name]
            for scenario_data in town_scenarios:

                if "scenario_type" not in scenario_data:
                    break
                scenario_name = scenario_data["scenario_type"]

                for scenario in scenario_data["available_event_configurations"]:
                    # Get the trigger point of the scenario
                    trigger_point = convert_dict_to_transform(scenario.pop('transform'))

                    # Check if the route passes through the scenario
                    match_position = RouteParser.match_trigger_to_route(trigger_point, trajectory)
                    if match_position is None:
                        continue

                    # Check the route has the correct topology
                    subtype = RouteParser.get_scenario_subtype(scenario_name, trajectory[match_position:])
                    if subtype is None:
                        continue

                    # Parse the scenario data
                    scenario_config = ScenarioConfiguration()
                    scenario_config.type = scenario_name
                    scenario_config.subtype = subtype
                    scenario_config.trigger_points = [trigger_point]
                    for other in scenario.pop('other_actors', []):
                        scenario_config.other_actors.append(ActorConfigurationData.parse_from_dict(other, 'scenario'))
                    scenario_config.other_parameters.update(scenario)

                    # Check if its location overlaps with other scenarios
                    existing_trigger = RouteParser.get_trigger_position(trigger_point, possible_scenarios.keys())
                    if existing_trigger:
                        possible_scenarios[existing_trigger].append(scenario_config)
                    else:
                        possible_scenarios.update({trigger_point: [scenario_config]})

        return possible_scenarios
