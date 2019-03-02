#!/usr/bin/env python
# Copyright (c) 2018-2019 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
CARLA Challenge Evaluator

Provisional code to evaluate Autonomous Agents for the CARLA Autonomous Driving challenge
"""

from __future__ import print_function
import argparse
from argparse import RawTextHelpFormatter
from datetime import datetime
import importlib
import numpy as np
import random
import sys
import time

import carla

from agents.navigation.local_planner import compute_connection, RoadOption
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from agents.tools.misc import vector

from srunner.challenge.envs.server_manager import ServerManagerBinary, ServerManagerDocker, Track
from srunner.challenge.envs.sensor_interface import CallBack, Speedometer
from srunner.scenarios.challenge_basic import *
from srunner.scenarios.config_parser import *
from srunner.scenariomanager.scenario_manager import ScenarioManager

# Dictionary of supported scenarios.
# key = Name of config file in configs/
# value = List as defined in the scenario module
SCENARIOS = {
    "ChallengeBasic": CHALLENGE_BASIC_SCENARIOS
}


class ChallengeEvaluator(object):

    """
    Provisional code to evaluate AutonomousAgent performance
    """

    ego_vehicle = None
    actors = []

    # Tunable parameters
    client_timeout = 15.0   # in seconds
    wait_for_world = 10.0  # in seconds

    # CARLA world and scenario handlers
    world = None
    manager = None

    def __init__(self, args):

        # first we instantiate the Agent
        module_name = os.path.basename(args.agent).split('.')[0]
        module_spec = importlib.util.spec_from_file_location(module_name, args.agent)
        self.module_agent = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(self.module_agent)

        self._sensors_list = []
        self._hop_resolution = 2.0

        # instantiate a CARLA server manager
        if args.use_docker:
            self._carla_server = ServerManagerDocker({'DOCKER_VERSION': args.docker_version})

        else:
            self._carla_server = ServerManagerBinary({'CARLA_SERVER': "{}/CarlaUE4.sh".format(args.carla_root)})


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

        # We need enumerate here, otherwise the actors are not properly removed
        for i, _ in enumerate(self.actors):
            if self.actors[i] is not None:
                self.actors[i].destroy()
                self.actors[i] = None
        self.actors = []

        for i, _ in enumerate(self._sensors_list):
            if self._sensors_list[i] is not None:
                self._sensors_list[i].destroy()
                self._sensors_list[i] = None
        self._sensors_list = []

        if ego and self.ego_vehicle is not None:
            self.ego_vehicle.destroy()
            self.ego_vehicle = None

    def setup_vehicle(self, model, spawn_point, hero=False, autopilot=False, random_location=False):
        """
        Function to setup the most relevant vehicle parameters,
        incl. spawn point and vehicle model.
        """

        blueprint_library = self.world.get_blueprint_library()

        # Get vehicle by model
        blueprint = random.choice(blueprint_library.filter(model))
        if hero:
            blueprint.set_attribute('role_name', 'hero')
        else:
            blueprint.set_attribute('role_name', 'scenario')

        if random_location:
            spawn_points = list(self.world.get_map().get_spawn_points())
            random.shuffle(spawn_points)
            for spawn_point in spawn_points:
                vehicle = self.world.try_spawn_actor(blueprint, spawn_point)
                if vehicle:
                    break
        else:
            vehicle = self.world.try_spawn_actor(blueprint, spawn_point)

        if vehicle is None:
            raise Exception(
                "Error: Unable to spawn vehicle {} at {}".format(model, spawn_point))
        else:
            # Let's deactivate the autopilot of the vehicle
            vehicle.set_autopilot(autopilot)

        return vehicle

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
            if sensor_spec['type'].startswith('sensor.speedometer'):
                # The speedometer pseudo sensor is created directly here
                sensor = Speedometer(vehicle, sensor_spec['reading_frequency'])
            # These are the sensors spawned on the carla world
            else:
                bp = bp_library.find(sensor_spec['type'])
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
                    bp.set_attribute('range', '200')
                    bp.set_attribute('rotation_frequency', '10')
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
            time.sleep(0.1)

    def prepare_actors(self, config):
        """
        Spawn or update all scenario actors according to
        their parameters provided in config
        """

        # If ego_vehicle already exists, just update location
        # Otherwise spawn ego vehicle
        if self.ego_vehicle is None:
            self.ego_vehicle = self.setup_vehicle(config.ego_vehicle.model, config.ego_vehicle.transform, hero=True)
        else:
            self.ego_vehicle.set_transform(config.ego_vehicle.transform)

        # setup sensors
        self.setup_sensors(self.agent_instance.sensors(), self.ego_vehicle)

        # spawn all other actors
        for actor in config.other_actors:
            new_actor = self.setup_vehicle(actor.model, actor.transform, hero=False, autopilot=actor.autopilot,
                                           random_location=actor.random_location)
            self.actors.append(new_actor)



    def analyze_scenario(self, args, config):
        """
        Provide feedback about success/failure of a scenario
        """
        result, score, return_message = self.manager.analyze_scenario_challenge()

        # show results stoud
        return_message_str = ""
        for msg in return_message:
            return_message_str += ("\n========== " + msg)
        report_string = "==[{}] [Score = {:.2f}] \n==[Comments:] {}".format(result, score, return_message_str)
        print(report_string)

        # save results in file
        current_time = str(datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
        filename = None
        if args.file:
            filename = config.name + current_time + ".txt"
            with open(filename, "a+") as fd:
                fd.write(report_string)

    def run(self, args):
        """
        Run all scenarios according to provided commandline args
        """

        # Prepare CARLA server
        self._carla_server.reset(args.host, args.port)
        self._carla_server.wait_until_ready()

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
                # create agent instance
                self.agent_instance = getattr(self.module_agent, self.module_agent.__name__)(args.config)

                # Prepare scenario
                print("Preparing scenario: " + config.name)
                scenario_class = ChallengeEvaluator.get_scenario_class_or_fail(config.type)

                client = carla.Client(args.host, int(args.port))
                client.set_timeout(self.client_timeout)

                # Once we have a client we can retrieve the world that is currently
                # running.
                self.world = client.load_world(config.town)

                # Wait for the world to be ready
                self.world.wait_for_tick(self.wait_for_world)

                # Create scenario manager
                self.manager = ScenarioManager(self.world, args.debug)

                try:
                    self.prepare_actors(config)
                    lat_ref, lon_ref = self._get_latlon_ref()
                    global_route, gps_route = self.retrieve_route(config.ego_vehicle, config.target, lat_ref, lon_ref)
                    config.route = global_route

                    scenario = scenario_class(self.world,
                                              self.ego_vehicle,
                                              self.actors,
                                              config.town,
                                              args.randomize,
                                              args.debug,
                                              config)
                except Exception as exception:
                    print("The scenario cannot be loaded")
                    print(exception)
                    self.cleanup(ego=True)
                    continue

                # Load scenario and run it
                self.manager.load_scenario(scenario)

                # debug
                waypoint_list, _ = zip(*global_route)
                self.draw_waypoints(waypoint_list, vertical_shift=1.0, persistency=scenario.timeout)
                # end debug

                self.manager.run_scenario(self.agent_instance)

                # Provide outputs if required
                self.analyze_scenario(args, config)

                # Stop scenario and cleanup
                self.manager.stop_scenario()
                del scenario

                self.cleanup(ego=True)
                self.agent_instance.destroy()

                # stop CARLA server
        self._carla_server.stop()

        print("No more scenarios .... Exiting")

    def draw_waypoints(self, waypoints, vertical_shift, persistency=-1):
        """
        Draw a list of waypoints at a certain height given in vertical_shift.

        :param waypoints: list or iterable container with the waypoints to draw
        :param vertical_shift: height in meters
        :return:
        """
        for w in waypoints:
            wp = w + carla.Location(z=vertical_shift)
            self.world.debug.draw_point(wp, size=0.1, color=carla.Color(0, 255, 0), life_time=persistency)


    def _get_latlon_ref(self):
        """
        Convert from waypoints world coordinates to CARLA GPS coordinates
        :return: tuple with lat and lon coordinates
        """
        xodr = self.world.get_map().to_opendrive()
        tree = ET.ElementTree(ET.fromstring(xodr))

        lat_ref = 0
        lon_ref = 0
        for opendrive in tree.iter("OpenDRIVE"):
            for header in opendrive.iter("header"):
                for georef in header.iter("geoReference"):
                    str_list = georef.text.split(' ')
                    lat_ref = float(str_list[0].split('=')[1])
                    lon_ref = float(str_list[1].split('=')[1])

        return lat_ref, lon_ref

    def retrieve_route(self, actor_configuration, target_configuration, lat_ref, lon_ref):
        """
        Estimate a route from a starting position to a target position.

        :param actor_configuration: ActorConfiguration object representing the ego vehicle
        :param target_configuration: TargetConfiguration object representing the target point
        :param lat_ref: map reference latitude
        :param lon_ref: map reference longitude
        :return: a tuple with two lists. Both lists represent the route.
            The first one in world coordinates.
            The second one in GPS coordinates.
        """
        start_waypoint = self.world.get_map().get_waypoint(actor_configuration.transform.location)
        end_waypoint = self.world.get_map().get_waypoint(target_configuration.transform.location)

        solution = []
        solution_gps = []

        # Setting up global router
        dao = GlobalRoutePlannerDAO(self.world.get_map())
        grp = GlobalRoutePlanner(dao)
        grp.setup()

        # Obtain route plan
        x1 = start_waypoint.transform.location.x
        y1 = start_waypoint.transform.location.y
        x2 = end_waypoint.transform.location.x
        y2 = end_waypoint.transform.location.y
        route = grp.plan_route((x1, y1), (x2, y2))

        current_waypoint = start_waypoint
        route.append(RoadOption.VOID)
        for action in route:

            #   Generate waypoints to next junction
            wp_choice = current_waypoint.next(self._hop_resolution)
            while len(wp_choice) == 1:
                current_waypoint = wp_choice[0]
                gps_point = self._location_to_gps(lat_ref, lon_ref, current_waypoint.transform.location)

                solution.append((current_waypoint.transform.location, RoadOption.LANEFOLLOW))
                solution_gps.append((gps_point, RoadOption.LANEFOLLOW))
                wp_choice = current_waypoint.next(self._hop_resolution)
                #   Stop at destination
                if current_waypoint.transform.location.distance(
                        end_waypoint.transform.location) < self._hop_resolution: break
            if action == RoadOption.VOID: break

            #   Select appropriate path at the junction
            if len(wp_choice) > 1:

                # Current heading vector
                current_transform = current_waypoint.transform
                current_location = current_transform.location
                projected_location = current_location + \
                                     carla.Location(
                                         x=math.cos(math.radians(
                                             current_transform.rotation.yaw)),
                                         y=math.sin(math.radians(
                                             current_transform.rotation.yaw)))
                v_current = vector(current_location, projected_location)

                direction = 0
                if action == RoadOption.LEFT:
                    direction = 1
                elif action == RoadOption.RIGHT:
                    direction = -1
                elif action == RoadOption.STRAIGHT:
                    direction = 0
                select_criteria = float('inf')

                #   Choose correct path
                for wp_select in wp_choice:
                    v_select = vector(
                        current_location, wp_select.transform.location)
                    cross = float('inf')
                    if direction == 0:
                        cross = abs(np.cross(v_current, v_select)[-1])
                    else:
                        cross = direction * np.cross(v_current, v_select)[-1]
                    if cross < select_criteria:
                        select_criteria = cross
                        current_waypoint = wp_select

                # Generate all waypoints within the junction
                #   along selected path
                gps_point = self._location_to_gps(lat_ref, lon_ref, current_waypoint.transform.location)
                solution.append((current_waypoint.transform.location, action))
                solution_gps.append((gps_point, action))
                current_waypoint = current_waypoint.next(self._hop_resolution)[0]
                while current_waypoint.is_intersection:
                    gps_point = self._location_to_gps(lat_ref, lon_ref, current_waypoint.transform.location)
                    solution.append((current_waypoint.transform.location, action))
                    solution_gps.append((gps_point, action))

                    current_waypoint = \
                        current_waypoint.next(self._hop_resolution)[0]

        # send plan to agent
        self.agent_instance.set_global_plan(solution_gps)

        return solution, solution_gps

    def _location_to_gps(self, lat_ref, lon_ref, location):
        """
        Convert from world coordinates to GPS coordinates
        :param lat_ref: latitude reference for the current map
        :param lon_ref: longitude reference for the current map
        :param location: location to translate
        :return: dictionary with lat, lon and height
        """
        EARTH_RADIUS_EQUA = 6378137.0

        scale = math.cos(lat_ref * math.pi / 180.0)
        mx = scale * lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
        my = scale * EARTH_RADIUS_EQUA * math.log(math.tan((90.0 + lat_ref) * math.pi / 360.0))
        mx += location.x
        my += location.y

        lon = mx * 180.0 / (math.pi * EARTH_RADIUS_EQUA * scale)
        lat = 360.0 * math.atan(math.exp(my / (EARTH_RADIUS_EQUA * scale))) / math.pi - 90.0
        z = location.z

        return {'lat':lat, 'lon':lon, 'z':z}


if __name__ == '__main__':

    DESCRIPTION = ("CARLA AD Challenge evaluation: evaluate your Agent in CARLA scenarios\n")

    PARSER = argparse.ArgumentParser(description=DESCRIPTION,
                                     formatter_class=RawTextHelpFormatter)
    PARSER.add_argument('--host', default='localhost',
                        help='IP of the host server (default: localhost)')
    PARSER.add_argument('--port', default='2000',
                        help='TCP port to listen to (default: 2000)')
    PARSER.add_argument("--use-docker", type=bool, help="Use docker to run CARLA?", default=False)
    PARSER.add_argument('--docker-version', type=str, help='Docker version to use for CARLA server', default="0.9.3")
    PARSER.add_argument("-a", "--agent", type=str, help="Path to Agent's py file to evaluate")
    PARSER.add_argument("--config", type=str, help="Path to Agent's configuration file", default="")

    PARSER.add_argument('--debug', action="store_true", help='Run with debug output')
    PARSER.add_argument('--file', action="store_true", help='Write results into a txt file')
    # pylint: disable=line-too-long
    PARSER.add_argument(
        '--scenario', help='Name of the scenario to be executed. Use the preposition \'group:\' to run all scenarios of one class, e.g. ControlLoss or FollowLeadingVehicle')
    # pylint: enable=line-too-long
    PARSER.add_argument('--randomize', action="store_true", help='Scenario parameters are randomized')
    PARSER.add_argument('--repetitions', default=1, help='Number of scenario executions')
    PARSER.add_argument('--list', action="store_true", help='List all supported scenarios and exit')
    PARSER.add_argument('--list_class', action="store_true", help='List all supported scenario classes and exit')
    ARGUMENTS = PARSER.parse_args()

    CARLA_ROOT = os.environ.get('CARLA_ROOT')
    ROOT_SCENARIO_RUNNER = os.environ.get('ROOT_SCENARIO_RUNNER')

    if not CARLA_ROOT:
        print("Error. CARLA_ROOT not found. Please run setup_environment.sh first.")
        sys.exit(0)

    if not ROOT_SCENARIO_RUNNER:
        print("Error. ROOT_SCENARIO_RUNNER not found. Please run setup_environment.sh first.")
        sys.exit(0)

    ARGUMENTS.carla_root = CARLA_ROOT

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

    try:
        challenge_evaluator = ChallengeEvaluator(ARGUMENTS)
        challenge_evaluator.run(ARGUMENTS)
    finally:
        del challenge_evaluator
