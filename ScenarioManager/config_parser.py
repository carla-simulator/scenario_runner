#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: Fabian Oboril (fabian.oboril@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
"""

from collections import namedtuple
import xml.etree.ElementTree as ET

import carla


class VehicleConfiguration(object):

    """
    """
    transform = None
    model = None

    def __init__(self, node):
        x = float(set_attrib(node, 'x', 0))
        y = float(set_attrib(node, 'y', 0))
        z = float(set_attrib(node, 'z', 0))
        yaw = float(set_attrib(node, 'yaw', 0))

        self.transform = carla.Transform(
            carla.Location(x=x, y=y, z=z), carla.Rotation(yaw=yaw))
        self.model = set_attrib(node, 'model', 'vehicle.*')


class ScenarioConfiguration(object):

    """
    """
    ego_vehicle = None
    other_vehicles = []
    town = None
    name = None


def set_attrib(node, key, default):
    return node.attrib[key] if node.attrib.has_key(key) else default


def prepare_scenario_actors(world, scenario_name):
    """
    """

    ego_vehicle_config = None
    other_vehicles = []

    scenario_config_file = "Configs/" + scenario_name + ".xml"
    tree = ET.parse(scenario_config_file)

    scenario_configurations = []

    for scenario in tree.iter("scenario"):

        new_config = ScenarioConfiguration()
        new_config.town = set_attrib(scenario, 'town', None)
        new_config.name = set_attrib(scenario, 'name', None)
        new_config.other_vehicles = []

        for ego_vehicle in scenario.iter("ego_vehicle"):
            new_config.ego_vehicle = VehicleConfiguration(ego_vehicle)

        for other_vehicle in scenario.iter("other_vehicle"):
            new_config.other_vehicles.append(
                VehicleConfiguration(other_vehicle))

        scenario_configurations.append(new_config)

    return scenario_configurations
