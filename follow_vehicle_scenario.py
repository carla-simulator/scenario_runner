#!/usr/bin/env python

#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Welcome to CARLA following vehicle scenario.

This is an example code on how to use a scenario, the scenario manager
and how to evaluate scenario results.
"""

import carla

from Scenarios.follow_leading_vehicle import FollowLeadingVehicle
from ScenarioManager.scenario_manager import ScenarioManager


def main():
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
        debug_mode = False
        scenario = FollowLeadingVehicle(world, debug_mode)
        manager = ScenarioManager(world, scenario, debug_mode)
        manager.run_scenario()

        if not manager.analyze_scenario():
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

    main()
