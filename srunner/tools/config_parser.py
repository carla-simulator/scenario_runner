#!/usr/bin/env python

# Copyright (c) 2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides a parser for scenario configuration files
"""

import glob
import os
import xml.etree.ElementTree as ET

import carla
from agents.navigation.local_planner import RoadOption


class RouteConfiguration(object):

    """
    This class provides the basic  configuration for a route
    """

    def __init__(self, route=None):
        if route is not None:
            self.data = route
        else:
            self.data = None

    def parse_xml(self, node):
        """
        Parse route config XML
        """
        self.data = []

        for waypoint in node.iter("waypoint"):
            x = float(set_attrib(waypoint, 'x', 0))
            y = float(set_attrib(waypoint, 'y', 0))
            z = float(set_attrib(waypoint, 'z', 0))
            c = set_attrib(waypoint, 'connection', '')
            connection = RoadOption[c.split('.')[1]]

            self.data.append((carla.Location(x, y, z), connection))


class TargetConfiguration(object):

    """
    This class provides the basic  configuration for a target location
    """

    transform = None

    def __init__(self, node):
        pos_x = float(set_attrib(node, 'x', 0))
        pos_y = float(set_attrib(node, 'y', 0))
        pos_z = float(set_attrib(node, 'z', 0))

        self.transform = carla.Transform(carla.Location(x=pos_x, y=pos_y, z=pos_z))


class ActorConfigurationData(object):

    """
    This is a configuration base class to hold model and transform attributes
    """

    def __init__(self, model, transform, autopilot=False, random=False, amount=1):
        self.model = model
        self.transform = transform
        self.autopilot = autopilot
        self.random_location = random
        self.amount = amount


class ActorConfiguration(ActorConfigurationData):

    """
    This class provides the basic actor configuration for a
    scenario:
    - Location and rotation (transform)
    - Model (e.g. Lincoln MKZ2017)
    """

    def __init__(self, node):

        pos_x = float(set_attrib(node, 'x', 0))
        pos_y = float(set_attrib(node, 'y', 0))
        pos_z = float(set_attrib(node, 'z', 0))
        yaw = float(set_attrib(node, 'yaw', 0))

        random_location = False
        if 'random_location' in node.keys():
            random_location = True

        autopilot = False
        if 'autopilot' in node.keys():
            autopilot = True

        amount = 1
        if 'amount' in node.keys():
            amount = int(node.attrib['amount'])

        super(ActorConfiguration, self).__init__(set_attrib(node, 'model', 'vehicle.*'),
                                                 carla.Transform(carla.Location(x=pos_x, y=pos_y, z=pos_z),
                                                 carla.Rotation(yaw=yaw)),
                                                 autopilot, random_location, amount)


class ScenarioConfiguration(object):

    """
    This class provides a basic scenario configuration incl.:
    - configurations for all actors
    - town, where the scenario should be executed
    - name of the scenario (e.g. ControlLoss_1)
    - type is the class of scenario (e.g. ControlLoss)
    """

    trigger_point = None
    ego_vehicle = None
    other_actors = []
    town = None
    name = None
    type = None
    target = None
    route = None


def set_attrib(node, key, default):
    """
    Parse XML key for a given node
    If key does not exist, use default value
    """
    return node.attrib[key] if key in node.attrib else default


def parse_scenario_configuration(file_name, scenario_name):
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
        file_name = scenario_name

    scenario_config_file = os.getenv('ROOT_SCENARIO_RUNNER', "./") + "/srunner/configs/" + file_name + ".xml"
    tree = ET.parse(scenario_config_file)

    scenario_configurations = []

    for scenario in tree.iter("scenario"):

        new_config = ScenarioConfiguration()
        new_config.town = set_attrib(scenario, 'town', None)
        new_config.name = set_attrib(scenario, 'name', None)
        new_config.type = set_attrib(scenario, 'type', None)
        new_config.other_actors = []

        for ego_vehicle in scenario.iter("ego_vehicle"):
            new_config.ego_vehicle = ActorConfiguration(ego_vehicle)
            new_config.trigger_point = new_config.ego_vehicle.transform

        for target in scenario.iter("target"):
            new_config.target = TargetConfiguration(target)

        for route in scenario.iter("route"):
            route_conf = RouteConfiguration()
            route_conf.parse_xml(route)
            new_config.route = route_conf

        for other_actor in scenario.iter("other_actor"):
            new_config.other_actors.append(ActorConfiguration(other_actor))

        if single_scenario_only:
            if new_config.name == scenario_name:
                scenario_configurations.append(new_config)
        else:
            scenario_configurations.append(new_config)

    return scenario_configurations


def get_list_of_scenarios():
    """
    Parse *all* config files and provide a list with all scenarios @return
    """

    list_of_config_files = glob.glob("{}/srunner/configs/*.xml".format(os.getenv('ROOT_SCENARIO_RUNNER', "./")))

    scenarios = []
    for file_name in list_of_config_files:
        tree = ET.parse(file_name)
        for scenario in tree.iter("scenario"):
            scenarios.append(set_attrib(scenario, 'name', None))

    return scenarios


def find_scenario_config(scenario_name):
    """
    Parse *all* config files and find first match for scenario config
    """

    list_of_config_files = glob.glob("{}/srunner/configs/*.xml".format(os.getenv('ROOT_SCENARIO_RUNNER', "./")))

    for file_name in list_of_config_files:
        tree = ET.parse(file_name)
        for scenario in tree.iter("scenario"):
            if set_attrib(scenario, 'name', None) == scenario_name:
                return os.path.basename(file_name).split(".")[0]

    return None
