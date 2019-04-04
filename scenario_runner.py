#!/usr/bin/env python

# Copyright (c) 2018-2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Welcome to CARLA scenario_runner

This is the main script to be executed when running a scenario.
It loads the scenario configuration, loads the scenario and manager,
and finally triggers the scenario execution.
"""

from __future__ import print_function

import argparse
import traceback
from argparse import RawTextHelpFormatter
from datetime import datetime

import carla

from srunner.scenariomanager.carla_data_provider import *
from srunner.scenariomanager.scenario_manager import ScenarioManager
from srunner.scenarios.background_activity import *
from srunner.scenarios.control_loss import *
from srunner.scenarios.follow_leading_vehicle import *
from srunner.scenarios.maneuver_opposite_direction import *
from srunner.scenarios.no_signal_junction_crossing import *
from srunner.scenarios.object_crash_intersection import *
from srunner.scenarios.object_crash_vehicle import *
from srunner.scenarios.opposite_vehicle_taking_priority import *
from srunner.scenarios.other_leading_vehicle import *
from srunner.scenarios.signalized_junction_left_turn import *
from srunner.scenarios.signalized_junction_right_turn import *
from srunner.tools.config_parser import *

# Version of scenario_runner
VERSION = 0.3


# Dictionary of all supported scenarios.
# key = Name of config file in Configs/
# value = List as defined in the scenario module
SCENARIOS = {
    "BackgroundActivity": BACKGROUND_ACTIVITY_SCENARIOS,
    "FollowLeadingVehicle": FOLLOW_LEADING_VEHICLE_SCENARIOS,
    "ObjectCrossing": OBJECT_CROSSING_SCENARIOS,
    "RunningRedLight": RUNNING_RED_LIGHT_SCENARIOS,
    "NoSignalJunction": NO_SIGNAL_JUNCTION_SCENARIOS,
    "VehicleTurning": VEHICLE_TURNING_SCENARIOS,
    "ControlLoss": CONTROL_LOSS_SCENARIOS,
    "OppositeDirection": MANEUVER_OPPOSITE_DIRECTION,
    "OtherLeadingVehicle": OTHER_LEADING_VEHICLE_SCENARIOS,
    "SignalizedJunctionRightTurn": TURNING_RIGHT_SIGNALIZED_JUNCTION_SCENARIOS,
    "SignalizedJunctionLeftTurn": TURN_LEFT_SIGNALIZED_JUNCTION_SCENARIOS
}


class ScenarioRunner(object):

    """
    This is the core scenario runner module. It is responsible for
    running (and repeating) a single scenario or a list of scenarios.

    Usage:
    scenario_runner = ScenarioRunner(args)
    scenario_runner.run(args)
    del scenario_runner
    """

    ego_vehicle = None

    # Tunable parameters
    client_timeout = 10.0  # in seconds
    wait_for_world = 10.0  # in seconds

    # CARLA world and scenario handlers
    world = None
    manager = None

    def __init__(self, args):
        """
        Setup CARLA client and world
        Setup ScenarioManager
        """

        # First of all, we need to create the client that will send the requests
        # to the simulator. Here we'll assume the simulator is accepting
        # requests in the localhost at port 2000.
        self.client = carla.Client(args.host, int(args.port))
        self.client.set_timeout(self.client_timeout)

        # Once we have a client we can retrieve the world that is currently
        # running.
        self.world = self.client.get_world()
        CarlaDataProvider.set_world(self.world)

    def __del__(self):
        """
        Cleanup and delete actors, ScenarioManager and CARLA world
        """

        self.cleanup(True)
        if self.manager is not None:
            del self.manager
        if self.world is not None:
            del self.world

    @staticmethod
    def get_scenario_class_or_fail(scenario):
        """
        Get scenario class by scenario name
        If scenario is not supported or not found, exit script
        """

        for scenarios in SCENARIOS.values():
            if scenario in scenarios:
                if scenario in globals():
                    return globals()[scenario]

        print("Scenario '{}' not supported ... Exiting".format(scenario))
        sys.exit(-1)

    def cleanup(self, ego=False):
        """
        Remove and destroy all actors
        """

        CarlaDataProvider.cleanup()
        CarlaActorPool.cleanup()

        if ego and self.ego_vehicle is not None:
            self.ego_vehicle.destroy()
            self.ego_vehicle = None

    def prepare_ego_vehicle(self, config):
        """
        Spawn or update the ego vehicle according to
        its parameters provided in config
        """

        # If ego_vehicle already exists, just update location
        # Otherwise spawn ego vehicle
        if self.ego_vehicle is None:
            self.ego_vehicle = CarlaActorPool.setup_actor(config.ego_vehicle.model,
                                                          config.ego_vehicle.transform,
                                                          True)
        else:
            self.ego_vehicle.set_transform(config.ego_vehicle.transform)

        # sync state
        CarlaDataProvider.get_world().wait_for_tick()

    def analyze_scenario(self, args, config):
        """
        Provide feedback about success/failure of a scenario
        """

        current_time = str(datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
        junit_filename = None
        if args.junit:
            junit_filename = config.name + current_time + ".xml"
        filename = None
        if args.file:
            filename = config.name + current_time + ".txt"

        if not self.manager.analyze_scenario(args.output, filename, junit_filename):
            print("Success!")
        else:
            print("Failure!")

    def run(self, args):
        """
        Run all scenarios according to provided commandline args
        """

        # Setup and run the scenarios for repetition times
        for _ in range(int(args.repetitions)):

            # Load the scenario configurations provided in the config file
            scenario_configurations = None
            if args.scenario.startswith("group:"):
                scenario_configurations = parse_scenario_configuration(args.scenario, args.scenario)
            else:
                scenario_config_file = find_scenario_config(args.scenario)
                if scenario_config_file is None:
                    print("Configuration for scenario {} cannot be found!".format(args.scenario))
                    continue
                scenario_configurations = parse_scenario_configuration(scenario_config_file, args.scenario)

            # Execute each configuration
            for config in scenario_configurations:
                self.world = self.client.load_world(config.town)
                CarlaActorPool.set_client(self.client)
                CarlaDataProvider.set_world(self.world)

                # Wait for the world to be ready
                self.world.wait_for_tick(self.wait_for_world)

                # Create scenario manager
                self.manager = ScenarioManager(self.world, args.debug)

                # Prepare scenario
                print("Preparing scenario: " + config.name)
                scenario_class = ScenarioRunner.get_scenario_class_or_fail(config.type)
                try:
                    CarlaActorPool.set_world(self.world)
                    self.prepare_ego_vehicle(config)
                    scenario = scenario_class(self.world,
                                              self.ego_vehicle,
                                              config,
                                              args.randomize,
                                              args.debug)
                except Exception as exception:
                    print("The scenario cannot be loaded")
                    if args.debug:
                        traceback.print_exc()
                    print(exception)
                    self.cleanup()
                    continue

                # Load scenario and run it
                self.manager.load_scenario(scenario)
                self.manager.run_scenario()

                # Provide outputs if required
                self.analyze_scenario(args, config)

                # Stop scenario and cleanup
                self.manager.stop_scenario()
                del scenario

                self.cleanup()

            print("No more scenarios .... Exiting")


if __name__ == '__main__':

    DESCRIPTION = ("CARLA Scenario Runner: Setup, Run and Evaluate scenarios using CARLA\n"
                   "Current version: " + str(VERSION))

    PARSER = argparse.ArgumentParser(description=DESCRIPTION,
                                     formatter_class=RawTextHelpFormatter)
    PARSER.add_argument('--host', default='127.0.0.1',
                        help='IP of the host server (default: localhost)')
    PARSER.add_argument('--port', default='2000',
                        help='TCP port to listen to (default: 2000)')
    PARSER.add_argument('--debug', action="store_true", help='Run with debug output')
    PARSER.add_argument('--output', action="store_true", help='Provide results on stdout')
    PARSER.add_argument('--file', action="store_true", help='Write results into a txt file')
    PARSER.add_argument('--junit', action="store_true", help='Write results into a junit file')
    # pylint: disable=line-too-long
    PARSER.add_argument(
        '--scenario', help='Name of the scenario to be executed. Use the preposition \'group:\' to run all scenarios of one class, e.g. ControlLoss or FollowLeadingVehicle')
    # pylint: enable=line-too-long
    PARSER.add_argument('--randomize', action="store_true", help='Scenario parameters are randomized')
    PARSER.add_argument('--repetitions', default=1, help='Number of scenario executions')
    PARSER.add_argument('--list', action="store_true", help='List all supported scenarios and exit')
    PARSER.add_argument('--list_class', action="store_true", help='List all supported scenario classes and exit')
    PARSER.add_argument('-v', '--version', action='version', version='%(prog)s ' + str(VERSION))
    ARGUMENTS = PARSER.parse_args()

    if ARGUMENTS.list:
        print("Currently the following scenarios are supported:")
        print(*get_list_of_scenarios(), sep='\n')
        sys.exit(0)

    if ARGUMENTS.list_class:
        print("Currently the following scenario classes are supported:")
        print(*SCENARIOS.keys(), sep='\n')
        sys.exit(0)

    if ARGUMENTS.scenario is None:
        print("Please specify a scenario using '--scenario SCENARIONAME'\n\n")
        PARSER.print_help(sys.stdout)
        sys.exit(0)

    SCENARIORUNNER = None
    try:
        SCENARIORUNNER = ScenarioRunner(ARGUMENTS)
        SCENARIORUNNER.run(ARGUMENTS)
    finally:
        if SCENARIORUNNER is not None:
            del SCENARIORUNNER
