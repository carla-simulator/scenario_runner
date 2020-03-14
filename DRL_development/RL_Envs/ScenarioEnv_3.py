"""
ps:

ScenarioEnv_3 is desired to be directly load scenario and train.


# ==================================================

statement on ScenarioEnv_3:

2020.02.24 liuyuqi

move ScenarioEnv_3 into scenario_runner folder,

using newest version of scenario_runner, updated on 2020.02.07

Ref on ScenarioEnv_1 and test_scenario and s2e

Merging different version of scenario_runner environment.

"""

import os
import sys
import glob

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

from __future__ import print_function

import traceback
import argparse
from argparse import RawTextHelpFormatter
from datetime import datetime
from distutils.version import LooseVersion
import argparse
import inspect
import threading
import signal
import importlib
import time
import math
import pkg_resources
import xml.etree.ElementTree as ET

# original import list of scenario_runner
from srunner.challenge.autoagents.agent_wrapper import SensorConfigurationInvalid
from srunner.challenge.challenge_statistics_manager import ChallengeStatisticsManager
from srunner.scenarioconfigs.openscenario_configuration import OpenScenarioConfiguration
from srunner.scenarioconfigs.route_scenario_configuration import RouteScenarioConfiguration
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider, CarlaActorPool
from srunner.scenariomanager.scenario_manager import ScenarioManager
# pylint: disable=unused-import
# For the following includes the pylint check is disabled, as these are accessed via globals()
from srunner.scenarios.control_loss import ControlLoss
from srunner.scenarios.follow_leading_vehicle import FollowLeadingVehicle, FollowLeadingVehicleWithObstacle
from srunner.scenarios.maneuver_opposite_direction import ManeuverOppositeDirection
from srunner.scenarios.no_signal_junction_crossing import NoSignalJunctionCrossing
from srunner.scenarios.object_crash_intersection import VehicleTurningRight, VehicleTurningLeft
from srunner.scenarios.object_crash_vehicle import StationaryObjectCrossing, DynamicObjectCrossing
from srunner.scenarios.opposite_vehicle_taking_priority import OppositeVehicleRunningRedLight
from srunner.scenarios.other_leading_vehicle import OtherLeadingVehicle
from srunner.scenarios.signalized_junction_left_turn import SignalizedJunctionLeftTurn
from srunner.scenarios.signalized_junction_right_turn import SignalizedJunctionRightTurn
from srunner.scenarios.change_lane import ChangeLane
from srunner.scenarios.cut_in import CutIn
# pylint: enable=unused-import
from srunner.scenarios.open_scenario import OpenScenario
from srunner.scenarios.route_scenario import RouteScenario
from srunner.tools.scenario_config_parser import ScenarioConfigurationParser
from srunner.tools.route_parser import RouteParser

# Version of scenario_runner
VERSION = 0.6

# self-added module to import
from srunner.scenariomanager.timer import GameTime
from srunner.challenge.envs.scene_layout_sensors import SceneLayoutReader, ObjectFinder
from srunner.challenge.envs.sensor_interface import CallBack, CANBusSensor, HDMapReader

from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from srunner.tools.scenario_helper import generate_target_waypoint
from srunner.challenge.utils.route_manipulation import interpolate_trajectory

# module to parser config
from srunner.tools.scenario_config_parser import ScenarioConfigurationParser


# import srunner.challenge.utils.route_configuration_parser as parser

# aviliable agent path
NPC_agent = "/home/lyq/Downloads/CARLA_096/scenario_runner/srunner/challenge/autoagents/NPCAgent.py"
Human_agent = "/home/lyq/Downloads/CARLA_096/scenario_runner/srunner/challenge/autoagents/HumanAgent.py"
Dummy_agent = "/home/lyq/Downloads/CARLA_096/scenario_runner/srunner/challenge/autoagents/DummyAgent.py"
# developing version
DRL_agent = "/home/lyq/Downloads/CARLA_096/scenario_runner/srunner/challenge/autoagents/DRL_development/DRLAgent_1.py"
agent_path = Dummy_agent

# scenario
scenario_class =

#
input_info = {
    'agent_path': agent_path,
    'scenario': scenario_class,

}

scenario_to_run = ''



class ScenarioTrainer(object):

    # client parameter
    host = 'localhost'
    port = 2000
    # Tunable parameters
    client_timeout = 30.0  # in seconds
    wait_for_world = 20.0  # in seconds

    MAX_ALLOWED_RADIUS_SENSOR = 5.0
    SECONDS_GIVEN_PER_METERS = 0.4
    MAX_CONNECTION_ATTEMPTS = 5



    def __init__(self, input_info):

        # setup world and client assuming that the CARLA server is up and running
        self.client = carla.Client(self.host, int(self.port))
        self.client.set_timeout(self.client_timeout)

        # parse config
        # manually set the scenario with name
        self.scenario = None




        #load world
        self.load_world()



        # create agent instance
        self.agent_instance = self.create_agent_instance(input_info['agent_path'])


        # the scenario is currently running and evaluating
        self.current_scenario = None
        self._sensors_list = []
        self.success_list = []

        self.debug = 0
        self.timestamp = None
        self.repetitions = 1

        # add collision check
        self.if_collision = False

        # all scenario instance list
        self.list_scenarios = []

        # repetitions
        # todo: use args to define
        self.repetitions = 1

    def load_world(self, client, town_name):

        self.world = client.load_world(town_name)
        self.timestamp = self.world.wait_for_tick(self.wait_for_world)
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        self.world.apply_settings(settings)

    def load_scenario(self):
        """
            Load scenario instance.

            Considered only 1 scenario to load,

            # todo: check which version of scenario_runner to use
            # currently use newest version
        """
        #
        scenario_config_file = ScenarioConfigurationParser.find_scenario_config(
            self.scenario, config_file_name = '')

        if scenario_config_file is None:
            print("Configuration for scenario {} cannot be found!".format(self._args.scenario))


        scenario_configurations = ScenarioConfigurationParser.parse_scenario_configuration(scenario_config_file,
                                                                                           self._args.scenario)

        # Execute each configuration
        for config in scenario_configurations:
            result = self._load_and_run_scenario(config)




        self.scenario =


    @staticmethod
    def print_avaliable_scenarios():
        """
            Print all supported scenarios.

            Ref on parser methods.
        """
        print("Currently the following scenarios are supported:")
        print(*ScenarioConfigurationParser.get_list_of_scenarios(config_file_name = ''), sep='\n')



        



    # todo: using import to load agent
    def create_agent_instance(self, agent_path):
        """
            Create the agent instance.
            Agent instance is seperated from ego vehicle.

        """
        # load agent from path
        module_name = os.path.basename(agent_path).split('.')[0]
        sys.path.insert(0, os.path.dirname(agent_path))
        self.module_agent = importlib.import_module(module_name)

        # todo: agent config is not used, remove it in future version
        agent_config = None
        self.agent_instance = getattr(self.module_agent, self.module_agent.__name__)(agent_config)


    def _cleanup(self, ego=False):
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

        if hasattr(self, '_sensors_list'):
            for i, _ in enumerate(self._sensors_list):
                if self._sensors_list[i] is not None:
                    self._sensors_list[i].stop()
                    self._sensors_list[i].destroy()
                    self._sensors_list[i] = None
            self._sensors_list = []
        if self.agent_instance is not None:
            self.agent_instance.destroy()

    def _analyze_scenario(self, args, config):
        """
        Return result of a scenario

        Result State is success or failure

        todo: Add a class attribute(or a list or dict) to store results

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
            return True
        else:
            print("Failure!")
            return False

    def _load_and_wait_for_world(self, args, town, ego_vehicles=None):
        """
        Load a new CARLA world and provide data to CarlaActorPool and CarlaDataProvider
        """

        if args.reloadWorld:
            self.world = self.client.load_world(town)
            settings = self.world.get_settings()
            settings.fixed_delta_seconds = 1.0 / self.frame_rate
            self.world.apply_settings(settings)

            # self.world = self.client.get_world()
            # settings = self.world.get_settings()
            # settings.synchronous_mode = False
            # settings.no_rendering_mode = False
            # settings.fixed_delta_seconds = 1.0 / self.frame_rate
            # self.world.apply_settings(settings)

            # self.world = self.client.load_world(town)
            # settings = self.world.get_settings()
            # settings.synchronous_mode = True
            # settings.fixed_delta_seconds = 1.0 / self.frame_rate
            # self.world.apply_settings(settings)

            self.world.on_tick(self._update_timestamp)
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
        # self.world.on_tick(self._update_timestamp)
        CarlaActorPool.set_client(self.client)
        CarlaActorPool.set_world(self.world)
        CarlaDataProvider.set_world(self.world)

        # Wait for the world to be ready
        self.world.tick()

        # todo: aotomated change map when map is wrong
        if CarlaDataProvider.get_map().name != town:
            print("The CARLA server uses the wrong map!")
            print("This scenario requires to use map {}".format(town))
            return False

        return True

    def _update_timestamp(self, snapshot):
        self.timestamp = snapshot.timestamp

    def setup_sensors(self, sensors, vehicle):
        """
        Create the sensors defined by the user and attach them to the ego-vehicle
        :param sensors: list of sensors
        :param vehicle: ego vehicle
        :return:
        """
        bp_library = self.world.get_blueprint_library()
        for sensor_spec in sensors:
            # These are the pseudosensors (not spawned)
            if sensor_spec['type'].startswith('sensor.scene_layout'):
                # Static sensor that gives you the entire information from the world (Just runs once)
                sensor = SceneLayoutReader(self.world)
            elif sensor_spec['type'].startswith('sensor.object_finder'):
                # This sensor returns the position of the dynamic objects in the scene.
                sensor = ObjectFinder(self.world, sensor_spec['reading_frequency'])
            elif sensor_spec['type'].startswith('sensor.can_bus'):
                # The speedometer pseudo sensor is created directly here
                sensor = CANBusSensor(vehicle, sensor_spec['reading_frequency'])
            elif sensor_spec['type'].startswith('sensor.hd_map'):
                # The HDMap pseudo sensor is created directly here
                sensor = HDMapReader(vehicle, sensor_spec['reading_frequency'])
            # These are the sensors spawned on the carla world
            else:
                bp = bp_library.find(str(sensor_spec['type']))
                if sensor_spec['type'].startswith('sensor.camera'):
                    bp.set_attribute('image_size_x', str(sensor_spec['width']))
                    bp.set_attribute('image_size_y', str(sensor_spec['height']))
                    bp.set_attribute('fov', str(sensor_spec['fov']))
                    sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                                     z=sensor_spec['z'])
                    sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                                     roll=sensor_spec['roll'],
                                                     yaw=sensor_spec['yaw'])
                elif sensor_spec['type'].startswith('sensor.lidar'):
                    bp.set_attribute('range', '5000')
                    bp.set_attribute('rotation_frequency', '20')
                    bp.set_attribute('channels', '32')
                    bp.set_attribute('upper_fov', '15')
                    bp.set_attribute('lower_fov', '-30')
                    bp.set_attribute('points_per_second', '500000')
                    sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                                     z=sensor_spec['z'])
                    sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                                     roll=sensor_spec['roll'],
                                                     yaw=sensor_spec['yaw'])
                elif sensor_spec['type'].startswith('sensor.other.gnss'):
                    sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                                     z=sensor_spec['z'])
                    sensor_rotation = carla.Rotation()

                # create sensor
                sensor_transform = carla.Transform(sensor_location, sensor_rotation)
                sensor = self.world.spawn_actor(bp, sensor_transform,
                                                vehicle)
            # setup callback
            sensor.listen(CallBack(sensor_spec['id'], sensor, self.agent_instance.sensor_interface))
            self._sensors_list.append(sensor)

        # check that all sensors have initialized their data structure

        while not self.agent_instance.all_sensors_ready():
            if self.debug > 0:
                print(" waiting for one data reading from sensors...")
            self.world.tick()

    # Take a deeper thinking on which structure of reset function should be construct ???
    # plan 1: packaging deleting scenario and actor together
    # plan 2: separate it in different module
    def reset(self):
        """
        Reset current scenario and rerun the simulation.

        Modified from challenge_evaluator_routes/run_route function

        Pipeline:

        Make judge if current scenario success or fail, if fail

        Delete actors in scenario (should world need reload ???)

        And Rebuild the scenario (Maybe call Initialization function )

        """

        # reset environment and simulation
        #
        # terminate behavior tree of scenario

        # delete actors

        # delete all scenario

        # check for scenario termination

        pass

    def scenario_is_running(self):
        """
            scenario_is_running

            Check if the scenario is running, basically a manual check

            Modified from challenge evaluator route_is_running function
        """
        if self.current_scenario is None:
            raise ValueError('No scenario is loaded and running')

        # The scenario status can be: INVALID, RUNNING, SUCCESS, FAILURE.
        # Only the last two indicate that the scenario was running but terminated
        # Therefore, return true when status is INVALID or RUNNING, false otherwise
        if (self.current_scenario.scenario.scenario_tree.status == py_trees.common.Status.RUNNING or
                self.current_scenario.scenario.scenario_tree.status == py_trees.common.Status.INVALID):
            return True
        else:
            return False

    def run_route(self):

        while self.scenario_is_running:
            GameTime.on_carla_tick(self.timestamp)
            CarlaDataProvider.on_carla_tick()
            # call function of instant
            ego_action = self.agent_instance()

            # Collision test
            # stop current scenario and re-run
            if self.agent_instance.collision_state == True:
                pass


            # apply control
            self.ego_vehicles[0].apply_control(ego_action)

            # tick scenario manually
            for scenario in self.list_scenarios:
                # most important line to tick, using py_trees bottom function
                scenario.scenario.scenario_tree.tick_once()
                # The scenarios may change the control if it applies.
                ego_action = scenario.change_control(ego_action)

                if self.debug > 1:
                    for actor in self.world.get_actors():
                        if 'vehicle' in actor.type_id or 'walker' in actor.type_id:
                            print(actor.get_transform())

            frame_number = 2
            for _ in range(frame_number):
                spectator = self.world.get_spectator()
                ego_trans = self.ego_vehicles[0].get_transform()
                angle = ego_trans.rotation.yaw
                d = 6.4
                a = math.radians(180+angle)
                location = carla.Location(d * math.cos(a), d * math.sin(a), 2.0) + ego_trans.location
                spectator.set_transform(carla.Transform(location, carla.Rotation(yaw=angle, pitch=-15)))
                # self.world.tick()

            # route visualization
            # if self.route_visible:
            #     turn_positions_and_labels = clean_route(trajectory)
            #     self.draw_waypoints(trajectory, turn_positions_and_labels,
            #                         vertical_shift=1.0, persistency=50000.0)
            #     self.route_visible = False

            # time continues, tick the world
            attempts = 0
            while attempts < self.MAX_CONNECTION_ATTEMPTS:
                try:
                    self.world.tick()
                    break
                except Exception:
                    attempts += 1
                    print('======[WARNING] The server is frozen [{}/{} attempts]!!'.format(attempts,
                                                                                           self.MAX_CONNECTION_ATTEMPTS))
                    time.sleep(2.0)
                    continue

            # check scenario termination
            for i, _ in enumerate(self.list_scenarios):
                # The scenario status can be: INVALID, RUNNING, SUCCESS, FAILURE. Only the last two
                # indiciate that the scenario was running but terminated
                # Remove the scenario when termination is clear --> not INVALID, not RUNNING
                if (self.list_scenarios[i].scenario.scenario_tree.status != py_trees.common.Status.RUNNING and
                        self.list_scenarios[i].scenario.scenario_tree.status != py_trees.common.Status.INVALID):
                    self.list_scenarios[i].scenario.terminate()
                    self.list_scenarios[i].remove_all_actors()
                    self.list_scenarios[i] = None
            self.list_scenarios[:] = [scenario for scenario in self.list_scenarios if scenario]


    def _load_and_run_scenario(self, args, config):
        """
        Load and run the scenario given by config
        """

        if not self._load_and_wait_for_world(args, config.town, config.ego_vehicles):
            self._cleanup()
            return

        # todo: repetitions can be set in args
        for repe_iter in range(int(self.repetitions)):

            # Prepare scenario
            print("Preparing scenario: " + config.name)
            print("Repetition number: " + str(repe_iter))
            # only use carla defined scenario, not OpenScenario
            try:
                CarlaActorPool.set_world(self.world)
                self._prepare_ego_vehicles(config.ego_vehicles, args.waitForEgo)
                scenario_class = self._get_scenario_class_or_fail(config.type)
                scenario = scenario_class(self.world,
                                          self.ego_vehicles,
                                          config,
                                          args.randomize,
                                          args.debug)
                self.current_scenario = scenario

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

            # ==================================================================================
            # scenario manager part
            # Create scenario manager
            # self.manager = ScenarioManager(self.world, args.debug)

            # Load scenario and run it
            # self.manager.load_scenario(scenario)
            # ==================================================================================

            # add spectator for visualization
            # todo: use a function or class to manage spectator
            spectator = self.world.get_spectator()
            ego_trans = self.ego_vehicles[0].get_transform()
            angle = ego_trans.rotation.yaw
            d = 6.4
            a = math.radians(180 + angle)
            location = carla.Location(d * math.cos(a), d * math.sin(a), 2.0) + ego_trans.location
            spectator.set_transform(carla.Transform(location, carla.Rotation(yaw=angle, pitch=-15)))

            # generate route
            gps_route, trajectory = generate_route(self.world, ego_trans.location)
            # set sensors and route
            if self.agent_instance is not None:
                self.agent_instance.set_global_plan(gps_route, trajectory)
                self.setup_sensors(self.agent_instance.sensors(), self.ego_vehicles[0])

            # if using waiting
            # todo: what's this for? Is it necessary?
            if args.waiting:
                try:
                    while True:
                        self.world.wait_for_tick()
                except:
                    print("\nCancel waiting, start running scenario")

            # run route
            self.run_route()


            # todo: check if success and reset

            # ==================================================================================
            # # Provide outputs if required
            # success = self._analyze_scenario(args, config)
            #
            # # use scenario manager to terminate scenario(behavior tree) and delete related actors
            # # Stop scenario and _cleanup
            # self.manager.stop_scenario()
            # scenario.remove_all_actors()
            # ==================================================================================


    def run_scenarios(self, args):
        """
        Modified from scenario_runner._run_scenarios function.

        Previous name: run_scenario_once

        To construct scenario and run selected scenario once

        todo: using dict or list to store scenarios that will be evaluated. i.e self.list_scenarios

        Single scenario is considered for now.

        """


        # todo: all scenario instance should be stored in the self.scenario_list
        self.scenario_list = None

        # scenario instance list
        for scenario_number, scenario_instance in enumerate(self.scenario_list):
            pass


        # iter loop for multiple repetition
        for repetition in range(self.repetitions):

            pass



        # load scenario class
        scenario_config_file = ScenarioConfigurationParser.find_scenario_config(args.scenario, args.configFile)
        if scenario_config_file is None:
            print("Configuration for scenario {} cannot be found!".format(args.scenario))
            return

        # read-in and parse config of scenarios
        scenario_configurations = ScenarioConfigurationParser.parse_scenario_configuration(scenario_config_file,
                                                                                           args.scenario)
        # Execute each configuration
        # only 1 config for now: 1 scene of 1 scenario class
        # todo: fix to suit more scenarios, use enumerate perhaps
        for config in scenario_configurations:
            #
            self._load_and_run_scenario(args, config)


        # todo: need to check
        self._cleanup(ego=(not args.waitForEgo))

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
        '--scenario', help='Name of the scenario to be executed. Use the preposition \'group:\' to run all scenarios of one class, e.g. ControlLoss or FollowLeadingVehicle',
        default="DynamicObjectCrossing_1")
    # pylint: enable=line-too-long
    PARSER.add_argument('--randomize', action="store_true", help='Scenario parameters are randomized')
    PARSER.add_argument('--repetitions', default=1, help='Number of scenario executions')
    PARSER.add_argument('--list', action="store_true", help='List all supported scenarios and exit')
    PARSER.add_argument('--listClass', action="store_true", help='List all supported scenario classes and exit')
    PARSER.add_argument('--openscenario', help='Provide an OpenSCENARIO definition')

    # ****************** add arguments to basic scenario_runner ******************
    PARSER.add_argument('--waiting', action="store_true", help='Just load scenario and then waiting')
    # PARSER.add_argument("-a", "--agent", type=str, help="Path to Agent's py file to evaluate",)
    PARSER.add_argument("--config", type=str, help="Path to Agent's configuration file", default="")
    # ****************************************************************************

    PARSER.add_argument('-v', '--version', action='version', version='%(prog)s ' + str(VERSION))
    ARGUMENTS = PARSER.parse_args()

    if ARGUMENTS.list:
        print("Currently the following scenarios are supported:")
        print(*ScenarioConfigurationParser.get_list_of_scenarios(ARGUMENTS.configFile), sep='\n')
        sys.exit(0)

    if ARGUMENTS.listClass:
        print("Currently the following scenario classes are supported:")
        print(*SCENARIOS.keys(), sep='\n')
        sys.exit(0)

    if not ARGUMENTS.scenario and not ARGUMENTS.openscenario:
        print("Please specify a scenario using '--scenario SCENARIONAME'\n\n")
        PARSER.print_help(sys.stdout)
        sys.exit(0)

    # cancel using args to load agent
    # if not ARGUMENTS.agent:
    #     print("Please specify an agent using '--agent PATH_TO_YOUR_AGENT'\n\n")
    #     PARSER.print_help(sys.stdout)
    #     sys.exit(0)

    # directly set agent path to args
    # NPC agent
    NPC_agent = "/home/lyq/Downloads/CARLA_096/scenario_runner/srunner/challenge/autoagents/NPCAgent.py"
    Human_agent = "/home/lyq/Downloads/CARLA_096/scenario_runner/srunner/challenge/autoagents/HumanAgent.py"
    Dummy_agent = "/home/lyq/Downloads/CARLA_096/scenario_runner/srunner/challenge/autoagents/DummyAgent.py"
    # developing version
    DRL_agent = "/home/lyq/Downloads/CARLA_096/scenario_runner/srunner/challenge/autoagents/DRL_development/DRLAgent_1.py"
    ARGUMENTS.agent = NPC_agent

    SCENARIOTRAINER = None
    try:
        SCENARIOTRAINER = ScenarioTrainer(ARGUMENTS)
        SCENARIOTRAINER.run_scenarios(ARGUMENTS)
    finally:
        if SCENARIOTRAINER is not None:
            del SCENARIOTRAINER

