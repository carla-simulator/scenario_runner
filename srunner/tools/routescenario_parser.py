#!/usr/bin/env python

# Copyright (c) 2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides a parser for scenario configurations that make use of routes to
connect several scenarios
"""

from __future__ import print_function

from srunner.tools.config_parser import ScenarioConfiguration


class RouteScenarioConfiguration(ScenarioConfiguration):

    """
    Basic configuration of a RouteScenario
    """

    def __init__(self, route_description, scenario_file):

        self.other_actors = []
        self.ego_vehicles = []
        self.trigger_points = []

        self.name = "RouteScenario"
        self.town = route_description['town_name']
        self.route_description = route_description

        self.scenario_file = scenario_file
