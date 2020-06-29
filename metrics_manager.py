#!/usr/bin/env python

# Copyright (c) 2020 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows the execution of user-implemented metrics

"""
Welcome to the ScenarioRunner's metric module

This is the main script to be executed when running a metric.
It is responsible of parsing all the information and executing
the metric specified by the user.
"""

import os
import sys
import importlib
import inspect
import json
import argparse
from argparse import RawTextHelpFormatter

import carla

from srunner.metrics.tools.metrics_parser import MetricsParser


class MetricsManager(object):
    """
    Main class of the metrics module. Handles the parsing and execution of
    the metrics.
    """

    def __init__(self, args):
        """
        Initialization of the metrics manager. This creates the client, needed to parse
        the information from the recorder, extract the metrics class, and runs it
        """

        self._args = args
        self._client = carla.Client(self._args.host, self._args.port)

        # Get the log information.
        recorder_file = "{}/{}".format(os.getenv('SCENARIO_RUNNER_ROOT', "./"), self._args.log)

        # client.start_recorder() doesn't check if the file is valid, so do so before calling it
        if recorder_file[-4:] != '.log':
            print("ERROR: The log argument has to point to a .log file")
            sys.exit(-1)
        if not os.path.exists(recorder_file):
            print("ERROR: The specified log file does not exist")
            sys.exit(-1)

        recorder_str = self._client.show_recorder_file_info(recorder_file, True)
        recorder_info = MetricsParser.parse_recorder_info(recorder_str)

        # Load the correct town and get its map (gives the user access to the map API)
        map_name = recorder_info[1][0]["map"]
        self._world = self._client.load_world(map_name)
        town_map = self._world.get_map()

        if self._args.criteria:
            with open(self._args.criteria) as criteria_file:
                criteria_dict = json.load(criteria_file)
        else:
            criteria_dict = None

        # Read and run the metric class
        self._metric_class = self._get_metric_class(self._args.metric)
        self._metric_class(town_map, recorder_info, criteria_dict)

    def _get_metric_class(self, metric_file):
        """
        Function to extract the metrics class from the path given by the metrics
        argument. Returns the first class found that is a child of BasicMetric

        Args:
            metric_file (str): path to the metric's file.
        """
        # Get their module
        module_name = os.path.basename(metric_file).split('.')[0]
        sys.path.insert(0, os.path.dirname(metric_file))
        metric_module = importlib.import_module(module_name)

        # And their members of type class
        for member in inspect.getmembers(metric_module, inspect.isclass):
            # Get the first one with parent BasicMetrics
            member_parent = member[1].__bases__[0]
            if 'BasicMetric' in str(member_parent):
                return member[1]

        print("No child class of BasicMetric was found ... Exiting")
        sys.exit(-1)

def main():
    """
    main function
    """

    # pylint: disable=line-too-long
    description = ("Scenario Runner's metrics module. Evaluate the execution of a specific scenario by developing your own metric.\n")

    parser = argparse.ArgumentParser(description=description,
                                    formatter_class=RawTextHelpFormatter)
    parser.add_argument('--host', default='127.0.0.1',
                        help='IP of the host server (default: localhost)')
    parser.add_argument('--port', '-p', default=2000,
                        help='TCP port to listen to (default: 2000)')
    parser.add_argument('--log', required=True,
                        help='Path to the CARLA recorder .log file (relative to SCENARIO_RUNNER_ROOT).\nThis file is created by the record functionality at ScenarioRunner')
    parser.add_argument('--metric', required=True,
                        help='Path to the .py file defining the used metric.\nSome examples at srunner/metrics')
    parser.add_argument('--criteria', default="",
                        help='Path to the .json file with the criteria information.\nThis file is created by the record functionality at ScenarioRunner')
    # pylint: enable=line-too-long

    args = parser.parse_args()

    MetricsManager(args)

if __name__ == "__main__":
    sys.exit(main())
