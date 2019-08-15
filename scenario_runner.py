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

import traceback
import argparse
from argparse import RawTextHelpFormatter
from datetime import datetime
from distutils.version import LooseVersion
import importlib
import inspect
import time
import pkg_resources

import carla

from srunner.scenariomanager.carla_data_provider import *
from srunner.scenariomanager.scenario_manager import ScenarioManager
from srunner.scenarios.background_activity import *
from srunner.scenarios.control_loss import *
from srunner.scenarios.follow_leading_vehicle import *
from srunner.scenarios.maneuver_opposite_direction import *
from srunner.scenarios.master_scenario import *
from srunner.scenarios.no_signal_junction_crossing import *
from srunner.scenarios.object_crash_intersection import *
from srunner.scenarios.object_crash_vehicle import *
from srunner.scenarios.opposite_vehicle_taking_priority import *
from srunner.scenarios.other_leading_vehicle import *
from srunner.scenarios.signalized_junction_left_turn import *
from srunner.scenarios.signalized_junction_right_turn import *
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.scenarios.open_scenario import OpenScenario
from srunner.tools.config_parser import *
from srunner.tools.openscenario_parser import OpenScenarioConfiguration

# Version of scenario_runner
VERSION = 0.5


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
    "SignalizedJunctionLeftTurn": TURN_LEFT_SIGNALIZED_JUNCTION_SCENARIOS,
    "MasterScenario": MASTER_SCENARIO
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

    ego_vehicles = []

    # Tunable parameters
    client_timeout = 30.0  # in seconds
    wait_for_world = 20.0  # in seconds
    frame_rate = 20.0      # in Hz

    # CARLA world and scenario handlers
    world = None
    manager = None

    additional_scenario_module = None

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

        dist = pkg_resources.get_distribution("carla")
        if LooseVersion(dist.version) < LooseVersion('0.9.6'):
            raise ImportError("CARLA version 0.9.6 or newer required. CARLA version found: {}".format(dist))

        # Load additional scenario definitions, if there are any
        if args.additionalScenario != '':
            module_name = os.path.basename(args.additionalScenario).split('.')[0]
            sys.path.insert(0, os.path.dirname(args.additionalScenario))
            self.additional_scenario_module = importlib.import_module(module_name)

    def __del__(self):
        """
        Cleanup and delete actors, ScenarioManager and CARLA world
        """

        self.cleanup(True)
        if self.manager is not None:
            del self.manager
        if self.world is not None:
            del self.world

    def get_scenario_class_or_fail(self, scenario):
        """
        Get scenario class by scenario name
        If scenario is not supported or not found, exit script
        """

        for scenarios in SCENARIOS.values():
            if scenario in scenarios:
                if scenario in globals():
                    return globals()[scenario]

        for member in inspect.getmembers(self.additional_scenario_module):
            if scenario in member and inspect.isclass(member[1]):
                return member[1]

        print("Scenario '{}' not supported ... Exiting".format(scenario))
        sys.exit(-1)

    def cleanup(self, ego=False):
        """
        Remove and destroy all actors
        """

        CarlaDataProvider.cleanup()
        CarlaActorPool.cleanup()

        for i, _ in enumerate(self.ego_vehicles):
            if self.ego_vehicles[i]:
                if ego:
                    self.ego_vehicles[i].destroy()
                self.ego_vehicles[i] = None
        self.ego_vehicles = []

    def prepare_ego_vehicles(self, config, wait_for_ego_vehicles=False):
        """
        Spawn or update the ego vehicle according to
        its parameters provided in config

        As the world is re-loaded for every scenario, no ego exists so far
        """

        if not wait_for_ego_vehicles:
            for vehicle in config.ego_vehicles:
                self.ego_vehicles.append(CarlaActorPool.setup_actor(vehicle.model,
                                                                    vehicle.transform,
                                                                    vehicle.rolename,
                                                                    True))
        else:
            ego_vehicle_missing = True
            while ego_vehicle_missing:
                self.ego_vehicles = []
                ego_vehicle_missing = False
                for ego_vehicle in config.ego_vehicles:
                    ego_vehicle_found = False
                    carla_vehicles = CarlaDataProvider.get_world().get_actors().filter('vehicle.*')
                    for carla_vehicle in carla_vehicles:
                        if carla_vehicle.attributes['role_name'] == ego_vehicle.rolename:
                            ego_vehicle_found = True
                            self.ego_vehicles.append(carla_vehicle)
                            break
                    if not ego_vehicle_found:
                        ego_vehicle_missing = True
                        break

            for i, _ in enumerate(self.ego_vehicles):
                self.ego_vehicles[i].set_transform(config.ego_vehicles[i].transform)

        # sync state
        CarlaDataProvider.get_world().tick()

    def analyze_scenario(self, args, config):
        """
        Provide feedback about success/failure of a scenario
        """

        current_time = str(datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
        junit_filename = None
        config_name = config.name
        if args.outputDir != '':
            config_name = os.path.join(args.outputDir, config_name)
        if args.junit:
            junit_filename = config_name + current_time + ".xml"
        filename = None
        if args.file:
            filename = config_name + current_time + ".txt"

        if not self.manager.analyze_scenario(args.output, filename, junit_filename):
            print("Success!")
        else:
            print("Failure!")

    def load_and_wait_for_world(self, args, config):
        """
        Load a new CARLA world and provide data to CarlaActorPool and CarlaDataProvider
        """

        if args.reloadWorld:
            self.world = self.client.load_world(config.town)
            settings = self.world.get_settings()
            settings.fixed_delta_seconds = 1.0 / self.frame_rate
            self.world.apply_settings(settings)
        else:
            # if the world should not be reloaded, wait at least until all ego vehicles are ready
            ego_vehicle_found = False
            if args.waitForEgo:
                while not ego_vehicle_found:
                    vehicles = self.client.get_world().get_actors().filter('vehicle.*')
                    for ego_vehicle in config.ego_vehicles:
                        ego_vehicle_found = False
                        for vehicle in vehicles:
                            if vehicle.attributes['role_name'] == ego_vehicle.rolename:
                                ego_vehicle_found = True
                                break
                        if not ego_vehicle_found:
                            print("Not all ego vehicles ready. Waiting ... ")
                            time.sleep(1)
                            break

        self.world = self.client.get_world()
        CarlaActorPool.set_client(self.client)
        CarlaActorPool.set_world(self.world)
        CarlaDataProvider.set_world(self.world)

        # Wait for the world to be ready
        self.world.tick()

        if CarlaDataProvider.get_map().name != config.town:
            print("The CARLA server uses the wrong map!")
            print("This scenario requires to use map {}".format(config.town))
            return False

        return True

    def load_and_run_scenario(self, args, config, scenario):
        """
        Load and run the given scenario
        """

        # Set the appropriate weather conditions
        weather = carla.WeatherParameters(
            cloudyness=config.weather.cloudyness,
            precipitation=config.weather.precipitation,
            precipitation_deposits=config.weather.precipitation_deposits,
            wind_intensity=config.weather.wind_intensity,
            sun_azimuth_angle=config.weather.sun_azimuth,
            sun_altitude_angle=config.weather.sun_altitude
        )

        self.world.set_weather(weather)

        # Load scenario and run it
        self.manager.load_scenario(scenario)
        self.manager.run_scenario()

        # Provide outputs if required
        self.analyze_scenario(args, config)

        # Stop scenario and cleanup
        self.manager.stop_scenario()
        scenario.remove_all_actors()

        self.cleanup()

    def run(self, args):
        """
        Run all scenarios according to provided commandline args
        """

        if args.openscenario:
            self.run_openscenario(args)
            return

        # Setup and run the scenarios for repetition times
        for _ in range(int(args.repetitions)):

            # Load the scenario configurations provided in the config file
            scenario_configurations = None
            scenario_config_file = find_scenario_config(args.scenario, args.configFile)
            if scenario_config_file is None:
                print("Configuration for scenario {} cannot be found!".format(args.scenario))
                continue

            scenario_configurations = parse_scenario_configuration(scenario_config_file, args.scenario)

            # Execute each configuration
            config_counter = 0
            for config in scenario_configurations:

                if not self.load_and_wait_for_world(args, config):
                    self.cleanup()
                    continue

                # Create scenario manager
                self.manager = ScenarioManager(self.world, args.debug)

                # Prepare scenario
                print("Preparing scenario: " + config.name)
                scenario_class = self.get_scenario_class_or_fail(config.type)
                try:
                    CarlaActorPool.set_world(self.world)
                    self.prepare_ego_vehicles(config, args.waitForEgo)
                    scenario = scenario_class(self.world,
                                              self.ego_vehicles,
                                              config,
                                              args.randomize,
                                              args.debug)
                except Exception as exception:
                    print("The scenario cannot be loaded")
                    if args.debug:
                        traceback.print_exc()
                    print(exception)
                    self.cleanup()
                    config_counter += 1
                    continue

                self.load_and_run_scenario(args, config, scenario)

                config_counter += 1

            self.cleanup(ego=(not args.waitForEgo))

            print("No more scenarios .... Exiting")

    def run_openscenario(self, args):
        """
        Run openscenario
        """

        # Load the scenario configurations provided in the config file
        if not os.path.isfile(args.openscenario):
            print("File does not exist")
            self.cleanup()
            return

        config = OpenScenarioConfiguration(args.openscenario)

        if not self.load_and_wait_for_world(args, config):
            self.cleanup()
            return

        # Create scenario manager
        self.manager = ScenarioManager(self.world, args.debug)

        # Prepare scenario
        print("Preparing scenario: " + config.name)
        try:
            CarlaActorPool.set_world(self.world)
            self.prepare_ego_vehicles(config, args.waitForEgo)
            scenario = OpenScenario(world=self.world,
                                    ego_vehicles=self.ego_vehicles,
                                    config=config,
                                    config_file=args.openscenario,
                                    timeout=100000)
        except Exception as exception:
            print("The scenario cannot be loaded")
            if args.debug:
                traceback.print_exc()
            print(exception)
            self.cleanup()
            return

        self.load_and_run_scenario(args, config, scenario)

        self.cleanup(ego=(not args.waitForEgo))

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
    PARSER.add_argument('--outputDir', default='', help='Directory for output files (default: this directory)')
    PARSER.add_argument('--waitForEgo', action="store_true", help='Connect the scenario to an existing ego vehicle')
    PARSER.add_argument('--configFile', default='', help='Provide an additional scenario configuration file (*.xml)')
    PARSER.add_argument('--additionalScenario', default='', help='Provide additional scenario implementations (*.py)')
    PARSER.add_argument('--reloadWorld', action="store_true",
                        help='Reload the CARLA world before starting a scenario (default=True)')
    # pylint: disable=line-too-long
    PARSER.add_argument(
        '--scenario', help='Name of the scenario to be executed. Use the preposition \'group:\' to run all scenarios of one class, e.g. ControlLoss or FollowLeadingVehicle')
    # pylint: enable=line-too-long
    PARSER.add_argument('--randomize', action="store_true", help='Scenario parameters are randomized')
    PARSER.add_argument('--repetitions', default=1, help='Number of scenario executions')
    PARSER.add_argument('--list', action="store_true", help='List all supported scenarios and exit')
    PARSER.add_argument('--listClass', action="store_true", help='List all supported scenario classes and exit')
    PARSER.add_argument('--openscenario', help='Provide an OpenScenario definition')
    PARSER.add_argument('-v', '--version', action='version', version='%(prog)s ' + str(VERSION))
    ARGUMENTS = PARSER.parse_args()

    if ARGUMENTS.list:
        print("Currently the following scenarios are supported:")
        print(*get_list_of_scenarios(ARGUMENTS.configFile), sep='\n')
        sys.exit(0)

    if ARGUMENTS.listClass:
        print("Currently the following scenario classes are supported:")
        print(*SCENARIOS.keys(), sep='\n')
        sys.exit(0)

    if not ARGUMENTS.scenario and not ARGUMENTS.openscenario:
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
