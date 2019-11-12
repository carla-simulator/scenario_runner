#!/usr/bin/env python

# Copyright (c) 2018-2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module contains a execution manager for the CARLA AD Benchmark
"""

from __future__ import print_function

import os
from collections import OrderedDict

from srunner.scenarioconfigs.route_scenario_configuration import RouteScenarioConfiguration
from srunner.scenariomanager.carla_data_provider import *
from srunner.tools.route_parser import RouteParser
import pdb

class ChallengeExecutionManager(object):
    """
    TODO: X
    """
    def __init__(self, routes_file, scenarios_file, challenge_phase, repetitions):
        self._phase = challenge_phase.split("_")[0]
        self._repetitions = 0
        self._routes_dict = OrderedDict()
        self._route_pointer = 0
        self._indices_to_keys = []

        scenario_runner_root = os.getenv('ROOT_SCENARIO_RUNNER', "./")
        weather_profiles = CarlaDataProvider.find_weather_presets()

        if self._phase == 'dev':
            routes_file = '{}/srunner/challenge/routes_devtest.xml'.format(scenario_runner_root)
            self._repetitions = 1
        elif self._phase == 'validation':
            routes_file = '{}/srunner/challenge/routes_testprep.xml'.format(scenario_runner_root)
            self._repetitions = 3
        elif self._phase == 'test':
            routes_file = '{}/srunner/challenge/routes_testchallenge.xml'.format(scenario_runner_root)
            self._repetitions = 3
        else:
            # debug mode: use provided routes_file
            self._repetitions = repetitions

        route_descriptions_list = RouteParser.parse_routes_file(routes_file)

        for _, route_description in enumerate(route_descriptions_list):
            for repetition in range(self._repetitions):
                profile = weather_profiles[repetition % len(weather_profiles)]

                route_config = RouteScenarioConfiguration(route_description, scenarios_file)
                route_config.weather = profile[0]
                route_config.weather.sun_azimuth = -1
                route_config.weather.sun_altitude = -1

                route_key = '{}/{}'.format(route_description['id'], repetition)
                self._routes_dict[route_key] = route_config
                self._indices_to_keys.append(route_key)

    def set_route(self, route_id, repetition):
        route_key = '{}/{}'.format(route_id, repetition)
        self._route_pointer = self._indices_to_keys.index(route_key)

    def peek_next_route(self):
        return self._route_pointer < len(self._routes_dict)

    def next_route(self):
        route_key = self._indices_to_keys[self._route_pointer]
        route_config = self._routes_dict[route_key]
        self._route_pointer = min(self._route_pointer + 1, len(self._routes_dict))
        repetition = int(route_key.split('/')[1])

        return route_config, repetition