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

from srunner.scenarioconfigs.scenario_configuration import ScenarioConfiguration, ActorConfiguration
from srunner.scenarioconfigs.route_scenario_configuration import RouteConfiguration, TargetConfiguration


class ScenarioConfigurationParser(object):

    """
    Pure static class providing access to parser methods for scenario configuration files (*.xml)
    """

    @staticmethod
    def parse_scenario_configuration(scenario_config_file, scenario_name):
        """
        Parse scenario configuration file and provide a list of
        ScenarioConfigurations @return

        If scenario_name starts with "group:" all scenarios within
        the config file will be returned. Otherwise only the scenario,
        that matches the scenario_name.
        """

        single_scenario_only = True
        if scenario_name.startswith("group:"):
            single_scenario_only = False
            scenario_name = scenario_name[6:]

        tree = ET.parse(scenario_config_file)

        scenario_configurations = []

        for scenario in tree.iter("scenario"):

            new_config = ScenarioConfiguration()
            new_config.town = scenario.attrib.get('town', None)
            new_config.name = scenario.attrib.get('name', None)
            new_config.type = scenario.attrib.get('type', None)
            new_config.other_actors = []
            new_config.ego_vehicles = []
            new_config.trigger_points = []

            for weather in scenario.iter("weather"):
                new_config.weather.cloudiness = float(weather.attrib.get("cloudiness", 0))
                new_config.weather.precipitation = float(weather.attrib.get("precipitation", 0))
                new_config.weather.precipitation_deposits = float(weather.attrib.get("precipitation_deposits", 0))
                new_config.weather.wind_intensity = float(weather.attrib.get("wind_intensity", 0.35))
                new_config.weather.sun_azimuth = float(weather.attrib.get("sun_azimuth", 0.0))
                new_config.weather.sun_altitude = float(weather.attrib.get("sun_altitude", 15.0))

            for ego_vehicle in scenario.iter("ego_vehicle"):
                new_config.ego_vehicles.append(ActorConfiguration(ego_vehicle, 'hero'))
                new_config.trigger_points.append(new_config.ego_vehicles[-1].transform)

            for target in scenario.iter("target"):
                new_config.target = TargetConfiguration(target)

            for route in scenario.iter("route"):
                route_conf = RouteConfiguration()
                route_conf.parse_xml(route)
                new_config.route = route_conf

            for other_actor in scenario.iter("other_actor"):
                new_config.other_actors.append(ActorConfiguration(other_actor, 'scenario'))

            if single_scenario_only:
                if new_config.name == scenario_name:
                    scenario_configurations.append(new_config)
            else:
                scenario_configurations.append(new_config)

        return scenario_configurations

    @staticmethod
    def get_list_of_scenarios(config_file_name):
        """
        Parse *all* config files and provide a list with all scenarios @return
        """

        list_of_config_files = glob.glob("{}/srunner/examples/*.xml".format(os.getenv('ROOT_SCENARIO_RUNNER', "./")))
        list_of_config_files += glob.glob("{}/srunner/examples/*.xosc".format(os.getenv('ROOT_SCENARIO_RUNNER', "./")))

        if config_file_name != '':
            list_of_config_files.append(config_file_name)

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

    @staticmethod
    def find_scenario_config(scenario_name, config_file_name):
        """
        Parse *all* config files and find first match for scenario config
        """

        list_of_config_files = glob.glob("{}/srunner/examples/*.xml".format(os.getenv('ROOT_SCENARIO_RUNNER', "./")))

        if config_file_name != '':
            list_of_config_files.append(config_file_name)

        if scenario_name.startswith("group:"):
            scenario_name = scenario_name[6:]

            for file_name in list_of_config_files:
                tree = ET.parse(file_name)
                for scenario in tree.iter("scenario"):
                    if scenario.attrib.get('type', None) == scenario_name:
                        return file_name

        else:
            for file_name in list_of_config_files:
                tree = ET.parse(file_name)
                for scenario in tree.iter("scenario"):
                    if scenario.attrib.get('name', None) == scenario_name:
                        return file_name

        return None
