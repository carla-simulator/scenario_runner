#!/usr/bin/env python

#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Welcome to CARLA following vehicle scenario.

This is an example code on how to use a scenario, the scenario manager
and how to evaluate scenario results.
"""

import argparse
from argparse import RawTextHelpFormatter

import carla

from Scenarios.follow_leading_vehicle import *
from ScenarioManager.scenario_manager import ScenarioManager


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
        for i in range(int(args.repetitions)):
            if args.scenario == "FollowLeadingVehicle":
                scenario = FollowLeadingVehicle(world, args.debug)
            elif args.scenario == "FollowLeadingVehicleWithObstacle":
                scenario = FollowLeadingVehicleWithObstacle(world, args.debug)
            else:
                raise Exception(
                    "Unsupported scenario with name: {}".format(args.scenario))
            manager.load_scenario(scenario)
            manager.run_scenario()

            junit_filename = None
            if args.junit is not None:
                junit_filename = args.junit.split(".")[0] + "_{}.xml".format(i)

            if not manager.analyze_scenario(args.output, args.filename, junit_filename):
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

    description = (
        "CARLA Scenario Runner: Setup, Run and Evaluate scenarios using CARLA\n")

    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=RawTextHelpFormatter)
    parser.add_argument('--host', default='localhost',
                        help='IP of the host server (default: localhost)')
    parser.add_argument('--port', default='2000',
                        help='TCP port to listen to (default: 2000)')
    parser.add_argument(
        '--debug', action="store_true", help='Run with debug output')
    parser.add_argument(
        '--output', action="store_true", help='Provide results on stdout')
    parser.add_argument('--filename', help='Write results into given file')
    parser.add_argument(
        '--junit', help='Write results into the given junit file')
    parser.add_argument('--scenario', required=True,
                        help='Name of the scenario to be executed')
    parser.add_argument(
        '--repetitions', default=1, help='Number of scenario executions')
    args = parser.parse_args()

    main(args)
