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
import os
import time
import pkg_resources

import carla

from srunner.challenge.autoagents.agent_wrapper import SensorConfigurationInvalid
from srunner.challenge.challenge_statistics_manager import ChallengeStatisticsManager
from srunner.scenarioconfigs.openscenario_configuration import OpenScenarioConfiguration
from srunner.scenarioconfigs.route_scenario_configuration import RouteScenarioConfiguration
from srunner.scenariomanager.carla_data_provider import *
from srunner.scenariomanager.scenario_manager import ScenarioManager
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
from srunner.scenarios.change_lane import *
from srunner.scenarios.cut_in import *
from srunner.scenarios.open_scenario import OpenScenario
from srunner.scenarios.route_scenario import RouteScenario
from srunner.tools.scenario_config_parser import ScenarioConfigurationParser
from srunner.tools.route_parser import RouteParser

# Version of scenario_runner
VERSION = 0.6


# Dictionary of all supported scenarios.
# key = Name of config file in examples/
# value = List as defined in the scenario module
SCENARIOS = {
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
    "ChangeLane": CHANGE_LANE_SCENARIOS,
    "CutIn": CUT_IN_SCENARIOS,
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

    agent_instance = None
    module_agent = None

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
        # If something goes wrong an exception will be thrown by importlib (ok here)
        if args.additionalScenario != '':
            module_name = os.path.basename(args.additionalScenario).split('.')[0]
            sys.path.insert(0, os.path.dirname(args.additionalScenario))
            self.additional_scenario_module = importlib.import_module(module_name)

        # Load agent if requested via command line args
        # If something goes wrong an exception will be thrown by importlib (ok here)
        if args.agent is not None:
            module_name = os.path.basename(args.agent).split('.')[0]
            sys.path.insert(0, os.path.dirname(args.agent))
            self.module_agent = importlib.import_module(module_name)

        # Create the ScenarioManager
        self.manager = ScenarioManager(args.debug, args.challenge)

        self._start_wall_time = datetime.now()

    def __del__(self):
        """
        Cleanup and delete actors, ScenarioManager and CARLA world
        """

        self._cleanup(True)
        if self.manager is not None:
            del self.manager
        if self.world is not None:
            del self.world

    def _within_available_time(self):
        """
        Check if the elapsed runtime is within the remaining user time budget
        Only relevant when running in challenge mode
        """
        current_time = datetime.now()
        elapsed_seconds = (current_time - self._start_wall_time).seconds

        return elapsed_seconds < os.getenv('CHALLENGE_TIME_AVAILABLE', '1080000')

    def _get_scenario_class_or_fail(self, scenario):
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

    def _cleanup(self, ego=False):
        """
        Remove and destroy all actors
        """

        self.client.stop_recorder()

        CarlaDataProvider.cleanup()
        CarlaActorPool.cleanup()

        for i, _ in enumerate(self.ego_vehicles):
            if self.ego_vehicles[i]:
                if ego:
                    self.ego_vehicles[i].destroy()
                self.ego_vehicles[i] = None
        self.ego_vehicles = []

        if self.agent_instance:
            self.agent_instance.destroy()
            self.agent_instance = None

    def _prepare_ego_vehicles(self, ego_vehicles, wait_for_ego_vehicles=False):
        """
        Spawn or update the ego vehicles
        """

        if not wait_for_ego_vehicles:
            for vehicle in ego_vehicles:
                self.ego_vehicles.append(CarlaActorPool.setup_actor(vehicle.model,
                                                                    vehicle.transform,
                                                                    vehicle.rolename,
                                                                    True,
                                                                    color=vehicle.color,
                                                                    vehicle_category=vehicle.category))
        else:
            ego_vehicle_missing = True
            while ego_vehicle_missing:
                self.ego_vehicles = []
                ego_vehicle_missing = False
                for ego_vehicle in ego_vehicles:
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
                self.ego_vehicles[i].set_transform(ego_vehicles[i].transform)

        # sync state
        CarlaDataProvider.get_world().tick()

    def _analyze_scenario(self, args, config):
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

    def _load_and_wait_for_world(self, args, town, ego_vehicles=None):
        """
        Load a new CARLA world and provide data to CarlaActorPool and CarlaDataProvider
        """

        if args.reloadWorld:
            self.world = self.client.load_world(town)
            settings = self.world.get_settings()
            settings.fixed_delta_seconds = 1.0 / self.frame_rate
            self.world.apply_settings(settings)
        else:
            # if the world should not be reloaded, wait at least until all ego vehicles are ready
            ego_vehicle_found = False
            if args.waitForEgo:
                while not ego_vehicle_found:
                    vehicles = self.client.get_world().get_actors().filter('vehicle.*')
                    for ego_vehicle in ego_vehicles:
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

        if args.agent:
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            self.world.apply_settings(settings)

        # Wait for the world to be ready
        if self.world.get_settings().synchronous_mode:
            self.world.tick()
        else:
            self.world.wait_for_tick()

        if CarlaDataProvider.get_map().name != town:
            print("The CARLA server uses the wrong map!")
            print("This scenario requires to use map {}".format(town))
            return False

        return True

    def _load_and_run_scenario(self, args, config):
        """
        Load and run the scenario given by config
        """

        if not self._load_and_wait_for_world(args, config.town, config.ego_vehicles):
            self._cleanup()
            return

        if args.agent:
            agent_class_name = self.module_agent.__name__.title().replace('_', '')
            try:
                self.agent_instance = getattr(self.module_agent, agent_class_name)(args.agentConfig)
                config.agent = self.agent_instance
            except Exception as e:
                print("Could not setup required agent due to {}".format(e))
                self._cleanup()
                return

        # Prepare scenario
        print("Preparing scenario: " + config.name)
        try:
            self._prepare_ego_vehicles(config.ego_vehicles, args.waitForEgo)
            if args.openscenario:
                scenario = OpenScenario(world=self.world,
                                        ego_vehicles=self.ego_vehicles,
                                        config=config,
                                        config_file=args.openscenario,
                                        timeout=100000)
            elif args.route:
                scenario = RouteScenario(world=self.world,
                                         config=config,
                                         debug_mode=args.debug)
            else:
                scenario_class = self._get_scenario_class_or_fail(config.type)
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
            self._cleanup()
            return

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

        # Set the appropriate road friction
        if config.friction is not None:
            friction_bp = self.world.get_blueprint_library().find('static.trigger.friction')
            extent = carla.Location(1000000.0, 1000000.0, 1000000.0)
            friction_bp.set_attribute('friction', str(config.friction))
            friction_bp.set_attribute('extent_x', str(extent.x))
            friction_bp.set_attribute('extent_y', str(extent.y))
            friction_bp.set_attribute('extent_z', str(extent.z))

            # Spawn Trigger Friction
            transform = carla.Transform()
            transform.location = carla.Location(-10000.0, -10000.0, 0.0)
            self.world.spawn_actor(friction_bp, transform)

        try:
            # Load scenario and run it
            if args.record:
                self.client.start_recorder("{}/{}.log".format(os.getenv('ROOT_SCENARIO_RUNNER', "./"), config.name))
            self.manager.load_scenario(scenario, self.agent_instance)
            self.manager.run_scenario()

            # Stop scenario
            self.manager.stop_scenario()

            # Provide outputs if required
            self._analyze_scenario(args, config)

            # Remove all actors
            scenario.remove_all_actors()
        except SensorConfigurationInvalid as e:
            self._cleanup(True)
            ChallengeStatisticsManager.record_fatal_error(e)
            sys.exit(-1)
        except Exception as e:
            if args.debug:
                traceback.print_exc()
            if args.challenge:
                ChallengeStatisticsManager.set_error_message(traceback.format_exc())
            print(e)

        self._cleanup()

    def _run_scenarios(self, args):
        """
        Run conventional scenarios (e.g. implemented using the Python API of ScenarioRunner)
        """

        # Setup and run the scenarios for repetition times
        for _ in range(int(args.repetitions)):

            # Load the scenario configurations provided in the config file
            scenario_configurations = None
            scenario_config_file = ScenarioConfigurationParser.find_scenario_config(args.scenario, args.configFile)
            if scenario_config_file is None:
                print("Configuration for scenario {} cannot be found!".format(args.scenario))
                continue

            scenario_configurations = ScenarioConfigurationParser.parse_scenario_configuration(scenario_config_file,
                                                                                               args.scenario)

            # Execute each configuration
            for config in scenario_configurations:
                self._load_and_run_scenario(args, config)

            self._cleanup(ego=(not args.waitForEgo))

            print("No more scenarios .... Exiting")

    def _run_challenge(self, args):
        """
        Run the challenge mode
        """

        phase_codename = os.getenv('CHALLENGE_PHASE_CODENAME', 'dev_track_3')
        phase = phase_codename.split("_")[0]

        repetitions = args.repetitions

        if args.challenge:
            weather_profiles = CarlaDataProvider.find_weather_presets()
            scenario_runner_root = os.getenv('ROOT_SCENARIO_RUNNER', "./")

            if phase == 'dev':
                routes = '{}/srunner/challenge/routes_devtest.xml'.format(scenario_runner_root)
                repetitions = 1
            elif phase == 'validation':
                routes = '{}/srunner/challenge/routes_testprep.xml'.format(scenario_runner_root)
                repetitions = 3
            elif phase == 'test':
                routes = '{}/srunner/challenge/routes_testchallenge.xml'.format(scenario_runner_root)
                repetitions = 3
            else:
                # debug mode
                routes = '{}/srunner/challenge/routes_debug.xml'.format(scenario_runner_root)
                repetitions = 1

        if args.route:
            routes = args.route[0]
            scenario_file = args.route[1]
            single_route = None
            if len(args.route) > 2:
                single_route = args.route[2]

        # retrieve routes
        route_descriptions_list = RouteParser.parse_routes_file(routes, single_route)
        # find and filter potential scenarios for each of the evaluated routes
        # For each of the routes and corresponding possible scenarios to be evaluated.
        if args.challenge:
            n_routes = len(route_descriptions_list) * repetitions
            ChallengeStatisticsManager.set_number_of_scenarios(n_routes)

        for _, route_description in enumerate(route_descriptions_list):
            for repetition in range(repetitions):

                if args.challenge and not self._within_available_time():
                    error_message = 'Not enough simulation time available to continue'
                    print(error_message)
                    ChallengeStatisticsManager.record_fatal_error(error_message)
                    self._cleanup(True)
                    sys.exit(-1)

                config = RouteScenarioConfiguration(route_description, scenario_file)

                if args.challenge:
                    profile = weather_profiles[repetition % len(weather_profiles)]
                    config.weather = profile[0]
                    config.weather.sun_azimuth = -1
                    config.weather.sun_altitude = -1

                self._load_and_run_scenario(args, config)
                self._cleanup(ego=(not args.waitForEgo))

    def _run_openscenario(self, args):
        """
        Run a scenario based on OpenSCENARIO
        """

        # Load the scenario configurations provided in the config file
        if not os.path.isfile(args.openscenario):
            print("File does not exist")
            self._cleanup()
            return

        config = OpenScenarioConfiguration(args.openscenario, self.client)
        self._load_and_run_scenario(args, config)
        self._cleanup(ego=(not args.waitForEgo))

    def run(self, args):
        """
        Run all scenarios according to provided commandline args
        """
        if args.openscenario:
            self._run_openscenario(args)
        elif args.route or args.challenge:
            self._run_challenge(args)
        else:
            self._run_scenarios(args)

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
    PARSER.add_argument('--randomize', action="store_true", help='Scenario parameters are randomized')
    PARSER.add_argument('--repetitions', default=1, help='Number of scenario executions')
    PARSER.add_argument('--list', action="store_true", help='List all supported scenarios and exit')
    PARSER.add_argument('--listClass', action="store_true", help='List all supported scenario classes and exit')
    PARSER.add_argument(
        '--agent', help="Agent used to execute the scenario (optional). Currently only compatible with route-based scenarios.")
    PARSER.add_argument('--agentConfig', type=str, help="Path to Agent's configuration file", default="")
    PARSER.add_argument('--openscenario', help='Provide an OpenSCENARIO definition')
    PARSER.add_argument(
        '--route', help='Run a route as a scenario, similar to the CARLA AD challenge (input: (route_file,scenario_file,[number of route]))', nargs='+', type=str)
    PARSER.add_argument('--challenge', action="store_true", help='Run in challenge mode')
    PARSER.add_argument('--record', action="store_true",
                        help='Use CARLA recording feature to create a recording of the scenario')
    PARSER.add_argument('-v', '--version', action='version', version='%(prog)s ' + str(VERSION))
    ARGUMENTS = PARSER.parse_args()
    # pylint: enable=line-too-long

    if ARGUMENTS.list:
        print("Currently the following scenarios are supported:")
        print(*ScenarioConfigurationParser.get_list_of_scenarios(ARGUMENTS.configFile), sep='\n')
        sys.exit(0)

    if ARGUMENTS.listClass:
        print("Currently the following scenario classes are supported:")
        print(*SCENARIOS.keys(), sep='\n')
        sys.exit(0)

    if not ARGUMENTS.scenario and not ARGUMENTS.openscenario and not ARGUMENTS.route:
        print("Please specify either a scenario or use the route mode\n\n")
        PARSER.print_help(sys.stdout)
        sys.exit(0)

    if (ARGUMENTS.route and ARGUMENTS.openscenario) or (ARGUMENTS.route and ARGUMENTS.scenario):
        print("The route mode cannot be used together with a scenario (incl. OpenSCENARIO)'\n\n")
        PARSER.print_help(sys.stdout)
        sys.exit(0)

    if ARGUMENTS.agent and (ARGUMENTS.openscenario or ARGUMENTS.scenario):
        print("Agents are currently only compatible with route scenarios'\n\n")
        PARSER.print_help(sys.stdout)
        sys.exit(0)

    if ARGUMENTS.challenge and (ARGUMENTS.openscenario or ARGUMENTS.scenario):
        print("The challenge mode can only be used with route-based scenarios'\n\n")
        PARSER.print_help(sys.stdout)
        sys.exit(0)

    if ARGUMENTS.route:
        ARGUMENTS.reloadWorld = True

    SCENARIORUNNER = None
    try:
        SCENARIORUNNER = ScenarioRunner(ARGUMENTS)
        SCENARIORUNNER.run(ARGUMENTS)
    finally:
        if ARGUMENTS.challenge:
            ChallengeStatisticsManager.report_challenge_statistics('results.json', ARGUMENTS.debug)
        if SCENARIORUNNER is not None:
            del SCENARIORUNNER
