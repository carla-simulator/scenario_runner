#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: Fabian Oboril (fabian.oboril@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Welcome to CARLA following vehicle scenario.

This is an example code on how to use a scenario, the scenario manager
and how to evaluate scenario results.
"""

from __future__ import print_function
import argparse
from argparse import RawTextHelpFormatter

import carla

from Scenarios.follow_leading_vehicle import *
from Scenarios.opposite_vehicle_taking_priority import *
from Scenarios.object_crash_vehicle import *
from Scenarios.no_signal_junction_crossing import NoSignalJunctionCrossing
from Scenarios.object_crash_intersection import *
from Scenarios.control_loss import *
from ScenarioManager.scenario_manager import ScenarioManager


# Version of scenario_runner
VERSION = 0.1


# List of all supported scenarios. IMPORTANT: String has to be class name
SCENARIOS = {
    "FollowLeadingVehicle",
    "FollowLeadingVehicleWithObstacle",
    "StationaryObjectCrossing",
    "DynamicObjectCrossing",
    "OppositeVehicleRunningRedLight",
    "NoSignalJunctionCrossing",
    "VehicleTurningRight",
    "VehicleTurningLeft",
    "ControlLoss"
}


def get_scenario_class_or_fail(scenario):
    """
    Get scenario class by scenario name
    If scenario is not supported or not found, raise an exception
    """
    if scenario in SCENARIOS:
        if scenario in globals():
            return globals()[scenario]
        else:
            raise Exception("No class for scenario '{}'".format(scenario))
    else:
        raise Exception("Scenario '{}' not supported".format(scenario))


def main(args):
    """
    Main function starting a CARLA client and connecting to the world.
    """

    # Tunable parameters
    client_timeout = 2.0   # in seconds
    wait_for_world = 10.0  # in seconds

    # CARLA world and scenario handlers
    world = None
    scenario = None
    manager = None

    try:
        # First of all, we need to create the client that will send the requests
        # to the simulator. Here we'll assume the simulator is accepting
        # requests in the localhost at port 2000.
        client = carla.Client(args.host, int(args.port))
        client.set_timeout(client_timeout)

        # Once we have a client we can retrieve the world that is currently
        # running.
        world = client.get_world()

        # Wait for the world to be ready
        world.wait_for_tick(wait_for_world)

        # Create scenario manager
        manager = ScenarioManager(world, args.debug)

        # Setup and run the scenario for repetition times
        scenario_class = get_scenario_class_or_fail(args.scenario)
        for i in range(int(args.repetitions)):
            scenario = scenario_class(world, args.debug)
            manager.load_scenario(scenario)
            manager.run_scenario()

            junit_filename = None
            if args.junit is not None:
                junit_filename = args.junit.split(".")[0] + "_{}.xml".format(i)

            if not manager.analyze_scenario(
                    args.output, args.filename, junit_filename):
                print("Success!")
            else:
                print("Failure!")

            manager.stop_scenario()
            del scenario

    finally:
        if manager is not None:
            del manager
        if world is not None:
            del world


if __name__ == '__main__':

    DESCRIPTION = (
        "CARLA Scenario Runner: Setup, Run and Evaluate scenarios using CARLA\n"
        "Current version: " + str(VERSION))

    PARSER = argparse.ArgumentParser(description=DESCRIPTION,
                                     formatter_class=RawTextHelpFormatter)
    PARSER.add_argument('--host', default='localhost',
                        help='IP of the host server (default: localhost)')
    PARSER.add_argument('--port', default='2000',
                        help='TCP port to listen to (default: 2000)')
    PARSER.add_argument(
        '--debug', action="store_true", help='Run with debug output')
    PARSER.add_argument(
        '--output', action="store_true", help='Provide results on stdout')
    PARSER.add_argument('--filename', help='Write results into given file')
    PARSER.add_argument(
        '--junit', help='Write results into the given junit file')
    PARSER.add_argument('--scenario',
                        help='Name of the scenario to be executed')
    PARSER.add_argument(
        '--repetitions', default=1, help='Number of scenario executions')
    PARSER.add_argument(
        '--list', action="store_true", help='List all supported scenarios and exit')
    PARSER.add_argument(
        '-v', '--version', action='version', version='%(prog)s ' + str(VERSION))
    ARGUMENTS = PARSER.parse_args()

    if ARGUMENTS.list:
        print("Currently the following scenarios are supported:")
        print(*SCENARIOS, sep='\n')
        sys.exit(0)

    if ARGUMENTS.scenario is None:
        print("Please specify a scenario using '--scenario SCENARIONAME'\n\n")
        PARSER.print_help(sys.stdout)
        sys.exit(0)

    main(ARGUMENTS)
