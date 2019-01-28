#!/usr/bin/env python

# Copyright (c) 2019 Intel Labs.
# authors: Fabian Oboril (fabian.oboril@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides a parser for scenario configuration files
"""

import xml.etree.ElementTree as ET

import carla


class ActorConfiguration(object):

    """
    This class provides the basic actor configuration for a
    scenario:
    - Location and rotation (transform)
    - Model (e.g. Lincoln MKZ2017)
    """

    transform = None
    model = None

    def __init__(self, node):
        pos_x = float(set_attrib(node, 'x', 0))
        pos_y = float(set_attrib(node, 'y', 0))
        pos_z = float(set_attrib(node, 'z', 0))
        yaw = float(set_attrib(node, 'yaw', 0))

        self.transform = carla.Transform(carla.Location(x=pos_x, y=pos_y, z=pos_z), carla.Rotation(yaw=yaw))
        self.model = set_attrib(node, 'model', 'vehicle.*')


class ScenarioConfiguration(object):

    """
    This class provides a basic scenario configuration incl.:
    - configurations for all actors
    - town, where the scenario should be executed
    - name of the scenario
    """

    ego_vehicle = None
    other_actors = []
    town = None
    name = None


def set_attrib(node, key, default):
    """
    Parse XML key for a given node
    If key does not exist, use default value
    """
    return node.attrib[key] if node.attrib.has_key(key) else default


def parse_scenario_configuration(world, file_name, scenario_name):
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

    scenario_config_file = "Configs/" + file_name + ".xml"
    tree = ET.parse(scenario_config_file)

    scenario_configurations = []

    for scenario in tree.iter("scenario"):

        new_config = ScenarioConfiguration()
        new_config.town = set_attrib(scenario, 'town', None)
        new_config.name = set_attrib(scenario, 'name', None)
        new_config.other_actors = []

        for ego_vehicle in scenario.iter("ego_vehicle"):
            new_config.ego_vehicle = ActorConfiguration(ego_vehicle)

        for other_actor in scenario.iter("other_actor"):
            new_config.other_actors.append(ActorConfiguration(other_actor))

        if single_scenario_only:
            if new_config.name == scenario_name:
                scenario_configurations.append(new_config)
        else:
            scenario_configurations.append(new_config)

    return scenario_configurations
