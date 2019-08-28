#!/usr/bin/env python

# Copyright (c) 2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides the key configuration parameters for a route-based scenario
"""

import carla
from agents.navigation.local_planner import RoadOption

from srunner.scenarioconfigs.scenario_configuration import ScenarioConfiguration


class RouteConfiguration(object):

    """
    This class provides the basic  configuration for a route
    """

    def __init__(self, route=None):
        self.data = route

    def parse_xml(self, node):
        """
        Parse route config XML
        """
        self.data = []

        for waypoint in node.iter("waypoint"):
            x = float(waypoint.attrib.get('x', 0))
            y = float(waypoint.attrib.get('y', 0))
            z = float(waypoint.attrib.get('z', 0))
            c = waypoint.attrib.get('connection', '')
            connection = RoadOption[c.split('.')[1]]

            self.data.append((carla.Location(x, y, z), connection))


class TargetConfiguration(object):

    """
    This class provides the basic configuration for a target location
    """

    transform = None

    def __init__(self, node):
        pos_x = float(node.attrib.get('x', 0))
        pos_y = float(node.attrib.get('y', 0))
        pos_z = float(node.attrib.get('z', 0))

        self.transform = carla.Transform(carla.Location(x=pos_x, y=pos_y, z=pos_z))


class RouteScenarioConfiguration(ScenarioConfiguration):

    """
    Basic configuration of a RouteScenario
    """

    def __init__(self, route_description, scenario_file):

        self.other_actors = []
        self.ego_vehicles = []
        self.trigger_points = []

        self.name = "RouteScenario_{}".format(route_description['id'])
        self.town = route_description['town_name']
        self.route_description = route_description

        self.scenario_file = scenario_file
