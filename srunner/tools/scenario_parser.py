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
from srunner.tools.carla_compat import IS_UE5


def _example_config_globs(extension):
    """Return the list of glob patterns for example config files of the given extension,
    in priority order (highest priority first).

    On UE5 servers (0.10.0+) we read configs from srunner/examples_ue5/ first so that
    scenarios authored for Town10HD_Opt take precedence over the legacy Town01-05
    entries when the same scenario name is defined in both locations. On 0.9.x only
    the legacy examples directory is read.
    """
    root = os.getenv('SCENARIO_RUNNER_ROOT', "./")
    patterns = []
    if IS_UE5:
        patterns.append("{}/srunner/examples_ue5/*.{}".format(root, extension))
    patterns.append("{}/srunner/examples/*.{}".format(root, extension))
    return patterns


def _collect_example_config_files(extensions):
    """Glob all example config files for the active engine version, in priority order.

    Each glob result is sorted so the within-directory walk is deterministic
    across platforms; the across-directory priority (examples_ue5 before
    examples on UE5) is preserved by the outer loop ordering.
    """
    files = []
    for ext in extensions:
        for pattern in _example_config_globs(ext):
            files.extend(sorted(glob.glob(pattern)))
    return files


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
        seen_names = set()

        list_of_config_files = _collect_example_config_files(["xml"])
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

                # Dedupe by scenario name. Files are walked in priority order
                # (examples_ue5 first on UE5), so the first occurrence wins.
                if scenario_config_name in seen_names:
                    continue
                seen_names.add(scenario_config_name)

                config = ScenarioConfiguration()
                config.town = scenario.attrib.get('town')
                config.name = scenario_config_name
                config.type = scenario_config_type

                for elem in list(scenario):
                    # Elements with special parsing
                    if elem.tag == 'ego_vehicle':
                        config.ego_vehicles.append(ActorConfigurationData.parse_from_node(elem, 'hero'))
                        config.trigger_points.append(config.ego_vehicles[-1].transform)
                    elif elem.tag == 'other_actor':
                        config.other_actors.append(ActorConfigurationData.parse_from_node(elem, 'scenario'))
                    elif elem.tag == 'weather':
                        for weather_attrib in elem.attrib:
                            if hasattr(config.weather, weather_attrib):
                                setattr(config.weather, weather_attrib, float(elem.attrib[weather_attrib]))
                            else:
                                print(f"WARNING: Ignoring '{weather_attrib}', as it isn't a weather parameter")

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
    def get_list_of_scenarios(additional_config_file_name):
        """
        Parse *all* config files and provide a list with all scenarios @return
        """

        list_of_config_files = _collect_example_config_files(["xml", "xosc"])
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
