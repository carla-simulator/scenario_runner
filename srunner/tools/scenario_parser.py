#!/usr/bin/env python

# Copyright (c) 2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides access to a scenario configuration parser
"""

import glob
import os
import xml.etree.ElementTree as ET

import carla

from srunner.scenarioconfigs.scenario_configuration import ScenarioConfiguration, ActorConfigurationData
from srunner.scenarioconfigs.route_scenario_configuration import RouteConfiguration


class ScenarioConfigurationParser(object):

    """
    Pure static class providing access to parser methods for scenario configuration files (*.xml)
    """

    @staticmethod
    def parse_scenario_configuration(scenario_name, additional_config_file_name):
        """
        Parse all scenario configuration files at srunner/examples and the additional
        config files, providing a list of ScenarioConfigurations @return

        If scenario_name starts with "group:" all scenarios that
        have that type are parsed and returned. Otherwise only the
        scenario that matches the scenario_name is parsed and returned.
        """

        if scenario_name.startswith("group:"):
            scenario_group = True
            scenario_name = scenario_name[6:]
        else:
            scenario_group = False

        scenario_configurations = []

        list_of_config_files = glob.glob("{}/srunner/examples/*.xml".format(os.getenv('SCENARIO_RUNNER_ROOT', "./")))
        if additional_config_file_name != '':
            list_of_config_files.append(additional_config_file_name)

        for file_name in list_of_config_files:
            tree = ET.parse(file_name)

            for scenario in tree.iter("scenario"):

                scenario_config_name = scenario.attrib.get('name', None)
                scenario_config_type = scenario.attrib.get('type', None)

                # Check that the scenario is the correct one
                if not scenario_group and scenario_config_name != scenario_name:
                    continue
                # Check that the scenario is of the correct type
                elif scenario_group and scenario_config_type != scenario_name:
                    continue

                config = ScenarioConfiguration()
                config.town = scenario.attrib.get('town')
                config.name = scenario_config_name
                config.type = scenario_config_type

                for elem in scenario.getchildren():
                    # Elements with special parsing
                    if elem.tag == 'ego_vehicle':
                        config.ego_vehicles.append(ActorConfigurationData.parse_from_node(elem, 'hero'))
                        config.trigger_points.append(config.ego_vehicles[-1].transform)
                    elif elem.tag == 'other_actor':
                        config.other_actors.append(ActorConfigurationData.parse_from_node(elem, 'scenario'))
                    elif elem.tag == 'weather':
                        config.weather.cloudiness = float(elem.attrib.get("cloudiness", 50))
                        config.weather.precipitation = float(elem.attrib.get("precipitation", 10))
                        config.weather.precipitation_deposits = float(elem.attrib.get("precipitation_deposits", 10))
                        config.weather.wind_intensity = float(elem.attrib.get("wind_intensity", 0.35))
                        config.weather.sun_azimuth_angle = float(elem.attrib.get("sun_azimuth_angle", 0.0))
                        config.weather.sun_altitude_angle = float(elem.attrib.get("sun_altitude_angle", 70.0))
                        config.weather.fog_density = float(elem.attrib.get("fog_density", 0.0))
                        config.weather.fog_distance = float(elem.attrib.get("fog_distance", 0.0))
                        config.weather.wetness = float(elem.attrib.get("wetness", 0.0))
                        config.weather.fog_falloff = float(elem.attrib.get("fog_falloff", 0.0))
                        config.weather.scattering_intensity = float(elem.attrib.get("scattering_intensity", 0.0))
                        config.weather.mie_scattering_scale = float(elem.attrib.get("mie_scattering_scale", 0.0))
                        config.weather.rayleigh_scattering_scale = float(
                            elem.attrib.get("rayleigh_scattering_scale", 0.0331))

                    elif elem.tag == 'route':
                        route_conf = RouteConfiguration()
                        route_conf.parse_xml(elem)
                        config.route = route_conf

                    # Any other possible element, add it as a config attribute
                    else:
                        config.other_parameters[elem.tag] = elem.attrib

                scenario_configurations.append(config)
        return scenario_configurations

    @staticmethod
    def parse_weather(weather_element):
        """Parses the weather element into a carla.WeatherParameters.
        """
        weather = carla.WeatherParameters()

        for weather_attrib in weather_element:

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

        return weather

    @staticmethod
    def get_list_of_scenarios(additional_config_file_name):
        """
        Parse *all* config files and provide a list with all scenarios @return
        """

        list_of_config_files = glob.glob("{}/srunner/examples/*.xml".format(os.getenv('SCENARIO_RUNNER_ROOT', "./")))
        list_of_config_files += glob.glob("{}/srunner/examples/*.xosc".format(os.getenv('SCENARIO_RUNNER_ROOT', "./")))
        if additional_config_file_name != '':
            list_of_config_files.append(additional_config_file_name)

        scenarios = []
        for file_name in list_of_config_files:
            if ".xosc" in file_name:
                tree = ET.parse(file_name)
                scenarios.append("{} (OpenSCENARIO)".format(tree.find("FileHeader").attrib.get('description', None)))
            else:
                tree = ET.parse(file_name)
                for scenario in tree.iter("scenario"):
                    scenarios.append(scenario.attrib.get('name', None))

        return scenarios
