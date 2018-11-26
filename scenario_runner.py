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

from Scenarios.follow_leading_vehicle import FollowLeadingVehicle
from ScenarioManager.scenario_manager import ScenarioManager


def main(stdout, filename, junit, scenario_name, debug_mode=False):
    """
    Main function starting a CARLA client and connecting to the world.
    """
    world = None
    scenario = None
    manager = None

    try:
        # First of all, we need to create the client that will send the requests
        # to the simulator. Here we'll assume the simulator is accepting
        # requests in the localhost at port 2000.
        client = carla.Client('localhost', 2000)
        client.set_timeout(2.0)

        # Once we have a client we can retrieve the world that is currently
        # running.
        world = client.get_world()

        # Wait for the world to be ready
        world.wait_for_tick(10.0)

        # Create scenario, manager and run scenario
        if scenario_name == "FollowLeadingVehicle":
            scenario = FollowLeadingVehicle(world, debug_mode)
        else:
            raise Exception(
                "Unsupported scenario with name: {}".format(scenario_name))
        manager = ScenarioManager(world, scenario, debug_mode)
        manager.run_scenario()

        if not manager.analyze_scenario(stdout, filename, junit):
            print("Success!")
        else:
            print("Failure!")

    finally:
        if manager is not None:
            manager.stop_scenario()
            del manager
        if scenario is not None:
            del scenario
        if world is not None:
            del world

if __name__ == '__main__':

    description = (
        "CARLA Scenario Runner: Setup, Run and Evaluate scenarios using CARLA\n")

    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        '--debug', action="store_true", help='Run with debug output')
    parser.add_argument(
        '--output', action="store_true", help='Provide results on stdout')
    parser.add_argument('--filename', help='Write results into given file')
    parser.add_argument(
        '--junit', help='Write results into the given junit file')
    parser.add_argument('--scenario', required=True,
                        help='Name of the scenario to be executed')
    args = parser.parse_args()

    main(args.output, args.filename, args.junit, args.scenario, args.debug)
