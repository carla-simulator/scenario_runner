#!/usr/bin/env python
# Copyright (c) 2018-2019 Intel Corporation.
# authors: German Ros (german.ros@intel.com), Felipe Codevilla (felipe.alcm@gmail.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
CARLA Challenge Evaluator Routes

Provisional code to evaluate Autonomous Agents for the CARLA Autonomous Driving challenge
"""
from __future__ import print_function
import argparse
from argparse import RawTextHelpFormatter
import copy
import datetime
import importlib
import math
import sys
import os
import json
import random
import re
import signal
import subprocess
import time
import traceback
import xml.etree.ElementTree as ET

import py_trees

import carla

from agents.navigation.local_planner import RoadOption

import srunner.challenge.utils.route_configuration_parser as parser
from srunner.challenge.envs.scene_layout_sensors import SceneLayoutReader, ObjectFinder
from srunner.challenge.envs.sensor_interface import CallBack, CANBusSensor, HDMapReader
from srunner.challenge.autoagents.autonomous_agent import Track
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.carla_data_provider import CarlaActorPool, CarlaDataProvider
from srunner.scenarios.control_loss import ControlLoss
from srunner.scenarios.background_activity import BackgroundActivity
from srunner.scenarios.follow_leading_vehicle import FollowLeadingVehicle
from srunner.scenarios.object_crash_vehicle import DynamicObjectCrossing
from srunner.scenarios.object_crash_intersection import VehicleTurningRight, VehicleTurningLeft
from srunner.scenarios.other_leading_vehicle import OtherLeadingVehicle
from srunner.scenarios.opposite_vehicle_taking_priority import OppositeVehicleRunningRedLight
from srunner.scenarios.signalized_junction_left_turn import SignalizedJunctionLeftTurn
from srunner.scenarios.signalized_junction_right_turn import SignalizedJunctionRightTurn
from srunner.scenarios.no_signal_junction_crossing import NoSignalJunctionCrossing
from srunner.scenarios.maneuver_opposite_direction import ManeuverOppositeDirection
from srunner.scenarios.master_scenario import MasterScenario
from srunner.scenarios.trafficlight_scenario import TrafficLightScenario
from srunner.challenge.utils.route_configuration_parser import TRIGGER_THRESHOLD, TRIGGER_ANGLE_THRESHOLD
from srunner.scenarioconfigs.scenario_configuration import ActorConfiguration, ScenarioConfiguration, ActorConfigurationData
from srunner.scenariomanager.traffic_events import TrafficEventType
from srunner.challenge.utils.route_manipulation import interpolate_trajectory, clean_route


number_class_translation = {

    "Scenario1": [ControlLoss],
    "Scenario2": [FollowLeadingVehicle],
    "Scenario3": [DynamicObjectCrossing],
    "Scenario4": [VehicleTurningRight, VehicleTurningLeft],
    "Scenario5": [OtherLeadingVehicle],
    "Scenario6": [ManeuverOppositeDirection],
    "Scenario7": [OppositeVehicleRunningRedLight],
    "Scenario8": [SignalizedJunctionLeftTurn],
    "Scenario9": [SignalizedJunctionRightTurn],
    "Scenario10": [NoSignalJunctionCrossing]

}

PENALTY_COLLISION_STATIC = 6
PENALTY_COLLISION_VEHICLE = 6
PENALTY_COLLISION_PEDESTRIAN = 9
PENALTY_TRAFFIC_LIGHT = 3
PENALTY_WRONG_WAY = 2
PENALTY_SIDEWALK_INVASION = 2
PENALTY_STOP = 2

# Util functions


def convert_json_to_actor(actor_dict):
    node = ET.Element('waypoint')
    node.set('x', actor_dict['x'])
    node.set('y', actor_dict['y'])
    node.set('z', actor_dict['z'])
    node.set('yaw', actor_dict['yaw'])

    return ActorConfiguration(node, rolename='simulation')


def convert_json_to_transform(actor_dict):

    return carla.Transform(location=carla.Location(x=float(actor_dict['x']), y=float(actor_dict['y']),
                                                   z=float(actor_dict['z'])),
                           rotation=carla.Rotation(roll=0.0, pitch=0.0, yaw=float(actor_dict['yaw'])))


def compare_scenarios(scenario_choice, existent_scenario):

    def transform_to_pos_vec(scenario):

        position_vec = [scenario['trigger_position']]
        if scenario['other_actors'] is not None:
            if 'left' in scenario['other_actors']:
                position_vec += scenario['other_actors']['left']
            if 'front' in scenario['other_actors']:
                position_vec += scenario['other_actors']['front']
            if 'right' in scenario['other_actors']:
                position_vec += scenario['other_actors']['right']

        return position_vec

    # put the positions of the scenario choice into a vec of positions to be able to compare

    choice_vec = transform_to_pos_vec(scenario_choice)
    existent_vec = transform_to_pos_vec(existent_scenario)
    for pos_choice in choice_vec:
        for pos_existent in existent_vec:

            dx = float(pos_choice['x']) - float(pos_existent['x'])
            dy = float(pos_choice['y']) - float(pos_existent['y'])
            dz = float(pos_choice['z']) - float(pos_existent['z'])
            dist_position = math.sqrt(dx * dx + dy * dy + dz * dz)
            dyaw = float(pos_choice['yaw']) - float(pos_choice['yaw'])
            dist_angle = math.sqrt(dyaw * dyaw)
            if dist_position < TRIGGER_THRESHOLD and dist_angle < TRIGGER_ANGLE_THRESHOLD:
                return True

    return False


def convert_transform_to_location(transform_vec):

    location_vec = []
    for transform_tuple in transform_vec:
        location_vec.append((transform_tuple[0].location, transform_tuple[1]))

    return location_vec


Z_DISTANCE_AVOID_COLLISION = 0.5  # z vallue to add in oder to avoid spawning vehicles to close to the ground


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


class ChallengeEvaluator(object):

    """
    Provisional code to evaluate AutonomousAgent performance
    """
    MAX_ALLOWED_RADIUS_SENSOR = 5.0
    SECONDS_GIVEN_PER_METERS = 0.4
    MAX_CONNECTION_ATTEMPTS = 5

    def __init__(self, args):
        phase_codename = os.getenv('CHALLENGE_PHASE_CODENAME', 'dev_track_3')
        if not phase_codename:
            raise ValueError('environment variable CHALLENGE_PHASE_CODENAME not defined')

        # retrieving scenario_runner root
        scenario_runner_root = os.getenv('ROOT_SCENARIO_RUNNER', '/workspace/scenario_runner')

        # remaining simulation time available for this time in seconds
        challenge_time_available = int(os.getenv('CHALLENGE_TIME_AVAILABLE', '1080000'))
        self.challenge_time_available = challenge_time_available

        self.start_wall_time = datetime.datetime.now()

        track = int(phase_codename.split("_")[2])
        phase_codename = phase_codename.split("_")[0]

        if phase_codename == 'dev':
            split_name = 'dev_split'
            self.routes = '{}/srunner/challenge/routes_devtest.xml'.format(scenario_runner_root)
            repetitions = 1
        elif phase_codename == 'validation':
            split_name = 'val_split'
            self.routes = '{}/srunner/challenge/routes_testprep.xml'.format(scenario_runner_root)
            repetitions = 3
        elif phase_codename == 'test':
            split_name = 'test_split'
            self.routes = '{}/srunner/challenge/routes_testchallenge.xml'.format(scenario_runner_root)
            repetitions = 3
        else:
            # debug mode
            # using short routes
            split_name = 'debug_split'
            self.routes = '{}/srunner/challenge/routes_debug.xml'.format(scenario_runner_root)
            repetitions = 1

        # overwriting routes in case users passed their own
        if args.routes:
            self.routes = args.routes

        if args.debug > 0:
            repetitions = 1

        if args.spectator:
            self.spectator = args.spectator
        else:
            self.spectator = False

        self.repetitions = repetitions
        self.phase = phase_codename
        self.split = split_name
        self.track = track
        self._error_message = ''

        self.debug = args.debug
        self.ego_vehicle = None
        self.actors = []
        self.statistics_routes = []
        self._current_route_broke = False
        self._system_error = False

        # Tunable parameters
        self.client_timeout = 30.0  # in seconds
        self.wait_for_world = 20.0  # in seconds
        self.frame_rate = 20.0      # in Hz

        # CARLA world and scenario handlers
        self.world = None
        self.agent_instance = None

        self.n_routes = 0
        self.weather_profiles = find_weather_presets()
        self.output_scenario = []
        self.master_scenario = None
        self.background_scenario = None
        self.list_scenarios = []

        # first we instantiate the Agent
        if args.agent is not None:
            module_name = os.path.basename(args.agent).split('.')[0]
            sys.path.insert(0, os.path.dirname(args.agent))
            self.module_agent = importlib.import_module(module_name)
        self._sensors_list = []
        self._hop_resolution = 2.0
        self.timestamp = None

        # debugging parameters
        self.route_visible = self.debug > 0

    def within_available_time(self):
        current_time = datetime.datetime.now()
        elapsed_seconds = (current_time - self.start_wall_time).seconds

        return elapsed_seconds < self.challenge_time_available

    def cleanup(self, ego=False):
        """
        Remove and destroy all actors
        """
        # We need enumerate here, otherwise the actors are not properly removed
        if hasattr(self, '_sensors_list'):
            for i, _ in enumerate(self._sensors_list):
                if self._sensors_list[i] is not None:
                    self._sensors_list[i].stop()
                    self._sensors_list[i].destroy()
                    self._sensors_list[i] = None
            self._sensors_list = []

        for i, _ in enumerate(self.actors):
            if self.actors[i] is not None:
                self.actors[i].destroy()
                self.actors[i] = None
        self.actors = []

        CarlaActorPool.cleanup()
        CarlaDataProvider.cleanup()

        if ego and self.ego_vehicle is not None:
            self.ego_vehicle.destroy()
            self.ego_vehicle = None

    def __del__(self):
        """
        Cleanup and delete actors, ScenarioManager and CARLA world
        """
        if hasattr(self, 'cleanup'):
            self.cleanup(True)
            if self.world is not None:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                self.world.apply_settings(settings)

                self.world = None

    def prepare_ego_car(self, start_transform):
        """
        Spawn or update all scenario actors according to
        a certain start position.
        """

        # If ego_vehicle already exists, just update location
        # Otherwise spawn ego vehicle
        if self.ego_vehicle is None:
            # TODO: the model is now hardcoded but that can change in a future.
            self.ego_vehicle = CarlaActorPool.request_new_actor('vehicle.lincoln.mkz2017',
                                                                start_transform,
                                                                rolename='hero',
                                                                hero=True)
        else:
            self.ego_vehicle.set_transform(start_transform)

        # setup sensors
        if self.agent_instance is not None:
            self.setup_sensors(self.agent_instance.sensors(), self.ego_vehicle)

    def draw_waypoints(self, waypoints, turn_positions_and_labels, vertical_shift, persistency=-1):
        """
        Draw a list of waypoints at a certain height given in vertical_shift.
        :param waypoints: list or iterable container with the waypoints to draw
        :param vertical_shift: height in meters
        :return:
        """
        for w in waypoints:
            wp = w[0].location + carla.Location(z=vertical_shift)
            self.world.debug.draw_point(wp, size=0.1, color=carla.Color(0, 255, 0), life_time=persistency)
        for start, end, conditions in turn_positions_and_labels:

            if conditions == RoadOption.LEFT:  # Yellow
                color = carla.Color(255, 255, 0)
            elif conditions == RoadOption.RIGHT:  # Cyan
                color = carla.Color(0, 255, 255)
            elif conditions == RoadOption.CHANGELANELEFT:  # Orange
                color = carla.Color(255, 64, 0)
            elif conditions == RoadOption.CHANGELANERIGHT:  # Dark Cyan
                color = carla.Color(0, 64, 255)
            else:  # STRAIGHT
                color = carla.Color(128, 128, 128)  # Gray

            for position in range(start, end):
                self.world.debug.draw_point(waypoints[position][0].location + carla.Location(z=vertical_shift),
                                            size=0.2, color=color, life_time=persistency)

        self.world.debug.draw_point(waypoints[0][0].location + carla.Location(z=vertical_shift), size=0.2,
                                    color=carla.Color(0, 0, 255), life_time=persistency)
        self.world.debug.draw_point(waypoints[-1][0].location + carla.Location(z=vertical_shift), size=0.2,
                                    color=carla.Color(255, 0, 0), life_time=persistency)

    def scenario_sampling(self, potential_scenarios_definitions):
        """
        The function used to sample the scenarios that are going to happen for this route.
        :param potential_scenarios_definitions: all the scenarios to be sampled
        :return: return the ones sampled for this case.
        """
        def position_sampled(scenario_choice, sampled_scenarios):
            # Check if this position was already sampled
            for existent_scenario in sampled_scenarios:
                # If the scenarios have equal positions then it is true.
                if compare_scenarios(scenario_choice, existent_scenario):
                    return True

            return False

        # The idea is to randomly sample a scenario per trigger position.
        sampled_scenarios = []
        for trigger in potential_scenarios_definitions.keys():
            possible_scenarios = potential_scenarios_definitions[trigger]
            scenario_choice = random.choice(possible_scenarios)
            del possible_scenarios[possible_scenarios.index(scenario_choice)]
            # We keep sampling and testing if this position is present on any of the scenarios.
            while position_sampled(scenario_choice, sampled_scenarios):
                if len(possible_scenarios) == 0:
                    scenario_choice = None
                    break
                scenario_choice = random.choice(possible_scenarios)
                del possible_scenarios[possible_scenarios.index(scenario_choice)]

            if scenario_choice is not None:
                sampled_scenarios.append(scenario_choice)

        return sampled_scenarios

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

    def get_actors_instances(self, list_of_antagonist_actors):
        """
        Get the full list of actor instances.
        """

        def get_actors_from_list(list_of_actor_def):
            """
                Receives a list of actor definitions and creates an actual list of ActorConfigurationObjects
            """
            sublist_of_actors = []
            for actor_def in list_of_actor_def:
                sublist_of_actors.append(convert_json_to_actor(actor_def))

            return sublist_of_actors

        list_of_actors = []
        # Parse vehicles to the left
        if 'front' in list_of_antagonist_actors:
            list_of_actors += get_actors_from_list(list_of_antagonist_actors['front'])

        if 'left' in list_of_antagonist_actors:
            list_of_actors += get_actors_from_list(list_of_antagonist_actors['left'])

        if 'right' in list_of_antagonist_actors:
            list_of_actors += get_actors_from_list(list_of_antagonist_actors['right'])

        return list_of_actors

    def build_master_scenario(self, route, town_name, timeout=300):
        # We have to find the target.
        # we also have to convert the route to the expected format
        master_scenario_configuration = ScenarioConfiguration()
        master_scenario_configuration.target = route[-1][0]  # Take the last point and add as target.
        master_scenario_configuration.route = convert_transform_to_location(route)
        master_scenario_configuration.town = town_name
        # TODO THIS NAME IS BIT WEIRD SINCE THE EGO VEHICLE  IS ALREADY THERE, IT IS MORE ABOUT THE TRANSFORM
        master_scenario_configuration.ego_vehicle = ActorConfigurationData('vehicle.lincoln.mkz2017',
                                                                           self.ego_vehicle.get_transform(),
                                                                           'hero')
        master_scenario_configuration.trigger_points = [self.ego_vehicle.get_transform()]
        CarlaDataProvider.register_actor(self.ego_vehicle)

        # Provide an initial blackboard entry to ensure all scenarios are running correctly
        blackboard = py_trees.blackboard.Blackboard()
        blackboard.set('master_scenario_command', 'scenarios_running')

        return MasterScenario(self.world, [self.ego_vehicle], master_scenario_configuration,
                              timeout=timeout, debug_mode=self.debug > 1)

    def build_background_scenario(self, town_name, timeout=300):
        scenario_configuration = ScenarioConfiguration()
        scenario_configuration.route = None
        scenario_configuration.town = town_name

        model = 'vehicle.*'
        transform = carla.Transform()
        rolename = 'background'
        autopilot = True
        random = True

        if town_name == 'Town01' or town_name == 'Town02':
            amount = 120
        elif town_name == 'Town03' or town_name == 'Town05':
            amount = 120
        elif town_name == 'Town04':
            amount = 200
        elif town_name == 'Town06' or town_name == 'Town07':
            amount = 150
        elif town_name == 'Town08':
            amount = 180
        elif town_name == 'Town09':
            amount = 350
        else:
            amount = 1

        actor_configuration_instance = ActorConfigurationData(model, transform, rolename, autopilot, random, amount)
        scenario_configuration.other_actors = [actor_configuration_instance]

        return BackgroundActivity(self.world, [self.ego_vehicle], scenario_configuration,
                                  timeout=timeout, debug_mode=self.debug > 1)

    def build_trafficlight_scenario(self, town_name, timeout=300):
        scenario_configuration = ScenarioConfiguration()
        scenario_configuration.route = None
        scenario_configuration.town = town_name

        return TrafficLightScenario(self.world, [self.ego_vehicle], scenario_configuration,
                                    timeout=timeout, debug_mode=self.debug > 1)

    def build_scenario_instances(self, scenario_definition_vec, town_name, timeout=300):
        """
            Based on the parsed route and possible scenarios, build all the scenario classes.
        :param scenario_definition_vec: the dictionary defining the scenarios
        :param town: the town where scenarios are going to be
        :return:
        """
        scenario_instance_vec = []

        for definition in scenario_definition_vec:
            # Get the class possibilities for this scenario number
            possibility_vec = number_class_translation[definition['name']]

            ScenarioClass = possibility_vec[definition['type']]

            # Create the other actors that are going to appear
            if definition['other_actors'] is not None:
                list_of_actor_conf_instances = self.get_actors_instances(definition['other_actors'])
            else:
                list_of_actor_conf_instances = []
            # Create an actor configuration for the ego-vehicle trigger position

            egoactor_trigger_position = convert_json_to_transform(definition['trigger_position'])
            scenario_configuration = ScenarioConfiguration()
            scenario_configuration.other_actors = list_of_actor_conf_instances
            scenario_configuration.town = town_name
            scenario_configuration.trigger_points = [egoactor_trigger_position]
            scenario_configuration.ego_vehicles = [ActorConfigurationData('vehicle.lincoln.mkz2017',
                                                                          self.ego_vehicle.get_transform(),
                                                                          'hero')]
            try:
                scenario_instance = ScenarioClass(self.world, [self.ego_vehicle], scenario_configuration,
                                                  criteria_enable=False, timeout=timeout)
            except Exception as e:
                if self.debug > 1:
                    raise e
                else:
                    print("Skipping scenario '{}' due to setup error: {}".format(definition['name'], e))
                    continue
            # registering the used actors on the data provider so they can be updated.

            CarlaDataProvider.register_actors(scenario_instance.other_actors)

            scenario_instance_vec.append(scenario_instance)

        return scenario_instance_vec

    def estimate_route_timeout(self, route):
        route_length = 0.0  # in meters

        prev_point = route[0][0]
        for current_point, _ in route[1:]:
            dist = current_point.location.distance(prev_point.location)
            route_length += dist
            prev_point = current_point

        return int(self.SECONDS_GIVEN_PER_METERS * route_length)

    def route_is_running(self):
        """
            The master scenario tests if the route is still running.
        """
        if self.master_scenario is None:
            raise ValueError('You should not run a route without a master scenario')

        # The scenario status can be: INVALID, RUNNING, SUCCESS, FAILURE. Only the last two
        # indiciate that the scenario was running but terminated
        # Therefore, return true when status is INVALID or RUNNING, false otherwise
        if (self.master_scenario.scenario.scenario_tree.status == py_trees.common.Status.RUNNING or
                self.master_scenario.scenario.scenario_tree.status == py_trees.common.Status.INVALID):
            return True
        else:
            return False

    def run_route(self, trajectory, no_master=False):
        while no_master or self.route_is_running():
            # update all scenarios
            GameTime.on_carla_tick(self.timestamp)
            CarlaDataProvider.on_carla_tick()
            # update all scenarios

            ego_action = self.agent_instance()
            for scenario in self.list_scenarios:
                scenario.scenario.scenario_tree.tick_once()
                # The scenarios may change the control if it applies.
                ego_action = scenario.change_control(ego_action)

                if self.debug > 1:
                    for actor in self.world.get_actors():
                        if 'vehicle' in actor.type_id or 'walker' in actor.type_id:
                            print(actor.get_transform())

            # ego vehicle acts
            self.ego_vehicle.apply_control(ego_action)
            if self.spectator:
                spectator = self.world.get_spectator()
                ego_trans = self.ego_vehicle.get_transform()
                spectator.set_transform(carla.Transform(ego_trans.location + carla.Location(z=50),
                                                        carla.Rotation(pitch=-90)))

                # show current score
                total_score, route_score, infractions_score = self.compute_current_statistics()
                print("{}/{}".format(route_score, infractions_score))
                self.world.debug.draw_string(ego_trans.location + carla.Location(z=15),
                                             "[{:.2f}/{:.2f}]".format(route_score, infractions_score),
                                             draw_shadow=False, color=carla.Color(255, 255, 255),
                                             life_time=0.01)

            if self.route_visible:
                turn_positions_and_labels = clean_route(trajectory)
                self.draw_waypoints(trajectory, turn_positions_and_labels,
                                    vertical_shift=1.0, persistency=50000.0)
                self.route_visible = False

            # time continues
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

            # check for scenario termination
            for i, _ in enumerate(self.list_scenarios):

                if self.debug == 1:
                    behavior = self.list_scenarios[i].scenario.scenario_tree.children[0]
                    if behavior.tip():
                        print("{} {} {} {}".format(self.list_scenarios[i].scenario.scenario_tree.name,
                                                   self.list_scenarios[i].scenario.scenario_tree.status,
                                                   behavior.tip().name,
                                                   behavior.tip().status))
                    if (behavior and behavior.tip() and behavior.tip().name !=
                        "InTriggerDistanceToLocationAlongRoute" and self.list_scenarios[
                        i].scenario.scenario_tree.name != "MasterScenario" and
                        self.list_scenarios[i].scenario.scenario_tree.name != "BackgroundActivity" and
                            self.list_scenarios[i].scenario.scenario_tree.name != "TrafficLightScenario"):
                        pass  # TODO: find a fix for this ascii bug
                        # py_trees.display.print_ascii_tree(self.list_scenarios[i].scenario.scenario_tree, 2, True)

                    # The scenario status can be: INVALID, RUNNING, SUCCESS, FAILURE. Only the last two
                    # indiciate that the scenario was running but terminated
                    # Remove the scenario when termination is clear --> not INVALID, not RUNNING
                if (self.list_scenarios[i].scenario.scenario_tree.status != py_trees.common.Status.RUNNING and
                        self.list_scenarios[i].scenario.scenario_tree.status != py_trees.common.Status.INVALID):
                    self.list_scenarios[i].scenario.terminate()
                    self.list_scenarios[i].remove_all_actors()
                    self.list_scenarios[i] = None
            self.list_scenarios[:] = [scenario for scenario in self.list_scenarios if scenario]

        # Route finished set for the background scenario to also finish
        blackboard = py_trees.blackboard.Blackboard()
        blackboard.set('master_scenario_command', 'scenarios_stop_request')

    def record_route_statistics(self, route_id):
        if self._current_route_broke:
            self.record_route_statistics_crash(route_id)
        else:
            self.record_route_statistics_default(route_id)

    def record_fatal_error(self, error_message):
        result = "ERROR"
        score_composed = 0.0
        score_penalty = 0.0
        score_route = 0.0

        return_message = error_message
        return_message += "\n=================================="

        current_statistics = {'id': -1,
                              'score_composed': score_composed,
                              'score_route': score_route,
                              'score_penalty': score_penalty,
                              'result': result,
                              'help_text': return_message
                              }

        self.statistics_routes.append(current_statistics)

    def record_route_statistics_crash(self, route_id):
        result = "CRASH"
        final_score = 0.0
        score_composed = 0.0
        score_penalty = 0.0
        score_route = 0.0

        return_message = ""
        return_message += "\n=================================="
        return_message += "\n==[r{}:{}] [Score = {:.2f} : (route_score={}, infractions=-{})]".format(route_id, result,
                                                                                                     final_score,
                                                                                                     score_route,
                                                                                                     score_penalty)
        return_message += "\n=================================="
        return_message += "\nCrash message: {}".format(self._error_message)
        return_message += "\n=================================="

        current_statistics = {'id': route_id,
                              'score_composed': score_composed,
                              'score_route': score_route,
                              'score_penalty': score_penalty,
                              'result': result,
                              'help_text': return_message
                              }

        self.statistics_routes.append(current_statistics)

    def compute_current_statistics(self):

        target_reached = False
        score_composed = 0.0
        score_penalty = 0.0
        score_route = 0.0

        list_traffic_events = []
        for node in self.master_scenario.scenario.test_criteria.children:
            if node.list_traffic_events:
                list_traffic_events.extend(node.list_traffic_events)

        list_collisions = []
        list_red_lights = []
        list_wrong_way = []
        list_route_dev = []
        list_sidewalk_inv = []
        list_stop_inf = []
        # analyze all traffic events
        for event in list_traffic_events:
            if event.get_type() == TrafficEventType.COLLISION_STATIC:
                score_penalty += PENALTY_COLLISION_STATIC
                msg = event.get_message()
                if msg:
                    list_collisions.append(event.get_message())

            elif event.get_type() == TrafficEventType.COLLISION_VEHICLE:
                score_penalty += PENALTY_COLLISION_VEHICLE
                msg = event.get_message()
                if msg:
                    list_collisions.append(event.get_message())

            elif event.get_type() == TrafficEventType.COLLISION_PEDESTRIAN:
                score_penalty += PENALTY_COLLISION_PEDESTRIAN
                msg = event.get_message()
                if msg:
                    list_collisions.append(event.get_message())

            elif event.get_type() == TrafficEventType.TRAFFIC_LIGHT_INFRACTION:
                score_penalty += PENALTY_TRAFFIC_LIGHT
                msg = event.get_message()
                if msg:
                    list_red_lights.append(event.get_message())

            elif event.get_type() == TrafficEventType.WRONG_WAY_INFRACTION:
                score_penalty += PENALTY_WRONG_WAY
                msg = event.get_message()
                if msg:
                    list_wrong_way.append(event.get_message())

            elif event.get_type() == TrafficEventType.ROUTE_DEVIATION:
                msg = event.get_message()
                if msg:
                    list_route_dev.append(event.get_message())

            elif event.get_type() == TrafficEventType.ON_SIDEWALK_INFRACTION:
                score_penalty += PENALTY_SIDEWALK_INVASION
                msg = event.get_message()
                if msg:
                    list_sidewalk_inv.append(event.get_message())

            elif event.get_type() == TrafficEventType.STOP_INFRACTION:
                score_penalty += PENALTY_STOP
                msg = event.get_message()
                if msg:
                    list_stop_inf.append(event.get_message())

            elif event.get_type() == TrafficEventType.ROUTE_COMPLETED:
                score_route = 100.0
                target_reached = True
            elif event.get_type() == TrafficEventType.ROUTE_COMPLETION:
                if not target_reached:
                    if event.get_dict():
                        score_route = event.get_dict()['route_completed']
                    else:
                        score_route = 0

        score_composed = max(score_route - score_penalty, 0.0)

        return score_composed, score_route, score_penalty

    def record_route_statistics_default(self, route_id):
        """
          This function is intended to be called from outside and provide
          statistics about the scenario (human-readable, for the CARLA challenge.)
        """

        target_reached = False
        failure = False
        result = "SUCCESS"
        score_composed = 0.0
        score_penalty = 0.0
        score_route = 0.0
        return_message = ""

        if self.master_scenario.scenario.test_criteria.status == py_trees.common.Status.FAILURE:
            failure = True
            result = "FAILURE"
        if self.master_scenario.scenario.timeout_node.timeout and not failure:
            result = "TIMEOUT"

        list_traffic_events = []
        for node in self.master_scenario.scenario.test_criteria.children:
            if node.list_traffic_events:
                list_traffic_events.extend(node.list_traffic_events)

        list_collisions = []
        list_red_lights = []
        list_wrong_way = []
        list_route_dev = []
        list_sidewalk_inv = []
        list_stop_inf = []
        # analyze all traffic events
        for event in list_traffic_events:
            if event.get_type() == TrafficEventType.COLLISION_STATIC:
                score_penalty += PENALTY_COLLISION_STATIC
                msg = event.get_message()
                if msg:
                    list_collisions.append(event.get_message())

            elif event.get_type() == TrafficEventType.COLLISION_VEHICLE:
                score_penalty += PENALTY_COLLISION_VEHICLE
                msg = event.get_message()
                if msg:
                    list_collisions.append(event.get_message())

            elif event.get_type() == TrafficEventType.COLLISION_PEDESTRIAN:
                score_penalty += PENALTY_COLLISION_PEDESTRIAN
                msg = event.get_message()
                if msg:
                    list_collisions.append(event.get_message())

            elif event.get_type() == TrafficEventType.TRAFFIC_LIGHT_INFRACTION:
                score_penalty += PENALTY_TRAFFIC_LIGHT
                msg = event.get_message()
                if msg:
                    list_red_lights.append(event.get_message())

            elif event.get_type() == TrafficEventType.WRONG_WAY_INFRACTION:
                score_penalty += PENALTY_WRONG_WAY
                msg = event.get_message()
                if msg:
                    list_wrong_way.append(event.get_message())

            elif event.get_type() == TrafficEventType.ROUTE_DEVIATION:
                msg = event.get_message()
                if msg:
                    list_route_dev.append(event.get_message())

            elif event.get_type() == TrafficEventType.ON_SIDEWALK_INFRACTION:
                score_penalty += PENALTY_SIDEWALK_INVASION
                msg = event.get_message()
                if msg:
                    list_sidewalk_inv.append(event.get_message())

            elif event.get_type() == TrafficEventType.STOP_INFRACTION:
                score_penalty += PENALTY_STOP
                msg = event.get_message()
                if msg:
                    list_stop_inf.append(event.get_message())

            elif event.get_type() == TrafficEventType.ROUTE_COMPLETED:
                score_route = 100.0
                target_reached = True
            elif event.get_type() == TrafficEventType.ROUTE_COMPLETION:
                if not target_reached:
                    if event.get_dict():
                        score_route = event.get_dict()['route_completed']
                    else:
                        score_route = 0

        score_composed = max(score_route - score_penalty, 0.0)

        return_message += "\n=================================="
        return_message += "\n==[r{}:{}] [Score = {:.2f} : (route_score={}, infractions=-{})]".format(route_id, result,
                                                                                                     score_composed,
                                                                                                     score_route,
                                                                                                     score_penalty)
        if list_collisions:
            return_message += "\n===== Collisions:"
            for item in list_collisions:
                return_message += "\n========== {}".format(item)

        if list_red_lights:
            return_message += "\n===== Red lights:"
            for item in list_red_lights:
                return_message += "\n========== {}".format(item)

        if list_stop_inf:
            return_message += "\n===== STOP infractions:"
            for item in list_stop_inf:
                return_message += "\n========== {}".format(item)

        if list_wrong_way:
            return_message += "\n===== Wrong way:"
            for item in list_wrong_way:
                return_message += "\n========== {}".format(item)

        if list_sidewalk_inv:
            return_message += "\n===== Sidewalk invasions:"
            for item in list_sidewalk_inv:
                return_message += "\n========== {}".format(item)

        if list_route_dev:
            return_message += "\n===== Route deviation:"
            for item in list_route_dev:
                return_message += "\n========== {}".format(item)

        return_message += "\n=================================="

        current_statistics = {'id': route_id,
                              'score_composed': score_composed,
                              'score_route': score_route,
                              'score_penalty': score_penalty,
                              'result': result,
                              'help_text': return_message
                              }

        self.statistics_routes.append(current_statistics)

    def report_challenge_statistics(self, filename, show_to_participant):
        score_composed = 0.0
        score_route = 0.0
        score_penalty = 0.0
        help_message = ""

        if self._system_error:
            submission_status = 'FAILED'

            for stats in self.statistics_routes:
                help_message += "{}\n\n".format(stats['help_text'])

        else:
            submission_status = 'FINISHED'

            for stats in self.statistics_routes:
                score_composed += stats['score_composed'] / float(self.n_routes)
                score_route += stats['score_route'] / float(self.n_routes)
                score_penalty += stats['score_penalty'] / float(self.n_routes)
                help_message += "{}\n\n".format(stats['help_text'])

            if self.debug > 0:
                print(help_message)

        # create json structure
        json_data = {
            'submission_status': submission_status,
            'stderr': help_message if self.phase == 'dev' or self.phase == 'debug' else 'No metadata provided for '
                                                                                         'this phase',
            'result': [
                {
                    'split': self.split,
                    'show_to_participant': show_to_participant,
                    'accuracies': {
                        'avg. route points': score_route,
                        'infraction points': score_penalty,
                        'total avg.': score_composed
                    }
                }],
            'metadata': [
                {
                    'stderr': help_message,
                    'accuracies': {
                        'avg. route points': score_route,
                        'infraction points': score_penalty,
                        'total avg.': score_composed
                    }
                }
            ]
        }

        with open(filename, "w+") as fd:
            fd.write(json.dumps(json_data, indent=4))

    def set_weather_profile(self, index):
        profile = self.weather_profiles[index % len(self.weather_profiles)]
        self.world.set_weather(profile[0])

    def load_world(self, client, town_name):
        # A new world can only be loaded in async mode
        if self.world is not None:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.no_rendering_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
        else:
            world = client.get_world()
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.no_rendering_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
            world = None

        self.world = client.load_world(town_name)
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / self.frame_rate
        if self.track == 4:
            settings.no_rendering_mode = True
        self.world.apply_settings(settings)

        self.world.on_tick(self._update_timestamp)

        # update traffic lights to make traffic more dynamic
        traffic_lights = self.world.get_actors().filter('*traffic_light*')
        for tl in traffic_lights:
            tl.set_green_time(9.0)
            tl.set_yellow_time(0.05)
            tl.set_red_time(0.08)

    def _update_timestamp(self, snapshot):
        self.timestamp = snapshot.timestamp

    def filter_scenarios(self, potential_scenarios_all, scenarios_to_remove):
        """

        :param potential_scenarios: the scenarios that we want to check
        :param scenarios_to_remove:  the scenarios that we still need to remove.
        :return: a list with the pontential scenarios without the scenarios to remove.
        """

        scenarios_after_filter = {}
        for trigger in potential_scenarios_all.keys():
            scenarios_after_filter.update({trigger: []})
            potential_scenarios_trigger = potential_scenarios_all[trigger]
            for possible_scenario in potential_scenarios_trigger:
                if possible_scenario['name'] not in set(scenarios_to_remove):
                    scenarios_after_filter[trigger].append(possible_scenario)
        return scenarios_after_filter

    def valid_sensors_configuration(self, agent, track):

        if self.phase != 'debug' and Track(track) != agent.track:
            return False, "You are submitting to the wrong track [{}]!".format(Track(track))

        sensors = agent.sensors()

        for sensor in sensors:
            if agent.track == Track.ALL_SENSORS:
                if sensor['type'].startswith('sensor.scene_layout') or sensor['type'].startswith(
                        'sensor.object_finder') or sensor['type'].startswith('sensor.hd_map'):
                    return False, "Illegal sensor used for Track [{}]!".format(agent.track)

            elif agent.track == Track.CAMERAS:
                if not (sensor['type'].startswith('sensor.camera.rgb') or sensor['type'].startswith(
                        'sensor.other.gnss') or sensor['type'].startswith('sensor.can_bus')):
                    return False, "Illegal sensor used for Track [{}]!".format(agent.track)

            elif agent.track == Track.ALL_SENSORS_HDMAP_WAYPOINTS:
                if sensor['type'].startswith('sensor.scene_layout') or sensor['type'].startswith('sensor.object_finder'):
                    return False, "Illegal sensor used for Track [{}]!".format(agent.track)
            else:
                if not (sensor['type'].startswith('sensor.scene_layout') or sensor['type'].startswith(
                        'sensor.object_finder') or sensor['type'].startswith('sensor.other.gnss')
                        or sensor['type'].startswith('sensor.can_bus')):
                    return False, "Illegal sensor used for Track [{}]!".format(agent.track)

            # let's check the extrinsics of the sensor
            if 'x' in sensor and 'y' in sensor and 'z' in sensor:
                if math.sqrt(sensor['x']**2 + sensor['y']**2 + sensor['z']**2) > self.MAX_ALLOWED_RADIUS_SENSOR:
                    return False, "Illegal sensor extrinsics used for Track [{}]!".format(agent.track)

        return True, ""

    def load_environment_and_run(self, args, world_annotations, route_description):

        # We need to copy the route_description to not override route_description
        _route_description = copy.copy(route_description)

        # prepare route's trajectory
        gps_route, _route_description['trajectory'] = interpolate_trajectory(self.world,
                                                                             _route_description['trajectory'])

        route_timeout = self.estimate_route_timeout(_route_description['trajectory'])

        potential_scenarios_definitions, _ = parser.scan_route_for_scenarios(_route_description,
                                                                             world_annotations)

        CarlaDataProvider.set_ego_vehicle_route(convert_transform_to_location(_route_description['trajectory']))

        # Sample the scenarios to be used for this route instance.
        sampled_scenarios_definitions = self.scenario_sampling(potential_scenarios_definitions)

        # create agent
        agent_class_name = self.module_agent.__name__.title().replace('_', '')
        self.agent_instance = getattr(self.module_agent, agent_class_name)(args.config)
        correct_sensors, error_message = self.valid_sensors_configuration(self.agent_instance, self.track)

        if not correct_sensors:
            # the sensor configuration is illegal
            self.record_fatal_error(error_message)
            self._system_error = True
            sys.exit(-1)

        self.agent_instance.set_global_plan(gps_route, _route_description['trajectory'])
        # prepare the ego car to run the route.
        # It starts on the first wp of the route

        elevate_transform = _route_description['trajectory'][0][0]
        elevate_transform.location.z += 0.5
        self.prepare_ego_car(elevate_transform)

        # build the master scenario based on the route and the target.
        self.master_scenario = self.build_master_scenario(_route_description['trajectory'],
                                                          _route_description['town_name'],
                                                          timeout=route_timeout)

        self.background_scenario = self.build_background_scenario(_route_description['town_name'],
                                                                  timeout=route_timeout)

        self.traffic_light_scenario = self.build_trafficlight_scenario(_route_description['town_name'],
                                                                       timeout=route_timeout)

        self.list_scenarios = [self.master_scenario, self.background_scenario, self.traffic_light_scenario]
        # build the instance based on the parsed definitions.
        if self.debug > 0:
            for scenario in sampled_scenarios_definitions:
                loc = carla.Location(scenario['trigger_position']['x'],
                                     scenario['trigger_position']['y'],
                                     scenario['trigger_position']['z']) + carla.Location(z=2.0)
                self.world.debug.draw_point(loc, size=1.0, color=carla.Color(255, 0, 0), life_time=100000)
                self.world.debug.draw_string(loc, str(scenario['name']), draw_shadow=False,
                                             color=carla.Color(0, 0, 255), life_time=100000, persistent_lines=True)

                print(scenario)

        self.list_scenarios += self.build_scenario_instances(sampled_scenarios_definitions,
                                                             _route_description['town_name'],
                                                             timeout=route_timeout)

        # Tick once to start the scenarios.
        if self.debug > 0:
            print(" Running these scenarios  --- ", self.list_scenarios)

        for scenario in self.list_scenarios:
            scenario.scenario.scenario_tree.tick_once()

        # main loop!
        self.run_route(_route_description['trajectory'])

    def run(self, args):
        """
        Run all routes according to provided commandline args
        """
        # do we have enough simulation time for this team?
        if not self.within_available_time():
            error_message = 'Not enough simulation time available to continue'
            self.record_fatal_error(error_message)
            self._system_error = True
            sys.exit(-1)

        # retrieve worlds annotations
        world_annotations = parser.parse_annotations_file(args.scenarios)
        # retrieve routes
        route_descriptions_list = parser.parse_routes_file(self.routes)
        # find and filter potential scenarios for each of the evaluated routes
        # For each of the routes and corresponding possible scenarios to be evaluated.
        self.n_routes = len(route_descriptions_list) * self.repetitions

        # setup world and client assuming that the CARLA server is up and running
        client = carla.Client(args.host, int(args.port))
        client.set_timeout(self.client_timeout)

        for route_idx, route_description in enumerate(route_descriptions_list):
            for repetition in range(self.repetitions):
                # check if we have enough wall time to run this specific route
                if not self.within_available_time():
                    error_message = 'Not enough simulation time available to run route [{}/{}]'.format(route_idx + 1,
                                                                                                       len(route_descriptions_list))
                    self.record_fatal_error(error_message)
                    self._system_error = True
                    sys.exit(-1)

                # For debugging
                self.route_visible = self.debug > 0

                # Try to load the world and start recording
                # If not successful stop recording and continue with next iteration
                try:
                    # load the self.world variable to be used during the route
                    self.load_world(client, route_description['town_name'])
                    # set weather profile
                    self.set_weather_profile(repetition)

                    # Set the actor pool so the scenarios can prepare themselves when needed
                    CarlaActorPool.set_client(client)
                    CarlaActorPool.set_world(self.world)
                    # Also se the Data provider pool.
                    CarlaDataProvider.set_world(self.world)
                    # tick world so we can start.
                    self.world.tick()

                    # start recording logs for the current route
                    client.start_recorder('log_{}_track{}_route_{:0>4d}.log'.format(self.phase, self.track, route_idx))
                except:
                    if args.show_to_participant:
                        self._error_message = traceback.format_exc()
                    client.stop_recorder()
                    continue

                # Try to run the route
                # If something goes wrong, still take the current score, and continue
                try:
                    self._current_route_broke = False
                    self.load_environment_and_run(args, world_annotations, route_description)
                except Exception as e:
                    if self.debug > 0:
                        traceback.print_exc()
                        raise
                    if self._system_error or not self.agent_instance:
                        print(e)
                        sys.exit(-1)
                    if args.show_to_participant:
                        self._error_message = traceback.format_exc()

                    self._current_route_broke = True

                # statistics recording
                self.record_route_statistics(route_description['id'])
                client.stop_recorder()

                # clean up
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                self.world.apply_settings(settings)
                self.agent_instance.destroy()
                self.agent_instance = None
                self.cleanup(ego=True)

                for scenario in self.list_scenarios:
                    # Reset scenario status for proper cleanup
                    scenario.scenario.terminate()
                    # Do not call del here! Directly enforce the actor removal
                    scenario.remove_all_actors()
                    scenario = None

                self.list_scenarios = []

                self.master_scenario = None
                self.background_scenario = None

                self.world.tick()

                if self.debug > 0:
                    break

        # final measurements from the challenge
        self.report_challenge_statistics(args.filename, args.show_to_participant)


if __name__ == '__main__':

    DESCRIPTION = ("CARLA AD Challenge evaluation: evaluate your Agent in CARLA scenarios\n")

    PARSER = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=RawTextHelpFormatter)
    PARSER.add_argument('--host', default='localhost',
                        help='IP of the host server (default: localhost)')
    PARSER.add_argument('--port', default='2000', help='TCP port to listen to (default: 2000)')
    PARSER.add_argument("-a", "--agent", type=str, help="Path to Agent's py file to evaluate")
    PARSER.add_argument("--config", type=str, help="Path to Agent's configuration file", default="")
    PARSER.add_argument('--debug', type=int, help='Run with debug output', default=0)
    PARSER.add_argument('--filename', type=str, help='Filename to store challenge results', default='results.json')
    PARSER.add_argument('--show-to-participant', type=bool, help='Show results to participant?', default=True)
    PARSER.add_argument('--spectator', type=bool, help='Switch spectator view on?', default=True)

    PARSER.add_argument('--routes',
                        help='Name of the route to be executed. Point to the route_xml_file to be executed.')
    PARSER.add_argument('--scenarios',
                        help='Name of the scenario annotation file to be mixed with the route.')

    ARGUMENTS = PARSER.parse_args()

    CARLA_ROOT = os.environ.get('CARLA_ROOT')
    ROOT_SCENARIO_RUNNER = os.environ.get('ROOT_SCENARIO_RUNNER')

    if not CARLA_ROOT:
        print("Error. CARLA_ROOT not found. Please run setup_environment.sh first.")
        sys.exit(0)

    if not ROOT_SCENARIO_RUNNER:
        print("Error. ROOT_SCENARIO_RUNNER not found. Please run setup_environment.sh first.")
        sys.exit(0)

    if ARGUMENTS.scenarios is None:
        print("Please specify a path to a scenario specification file  '--scenarios path-to-file'\n\n")
        PARSER.print_help(sys.stdout)
        sys.exit(0)

    ARGUMENTS.carla_root = CARLA_ROOT
    challenge_evaluator = None

    phase_codename = os.getenv('CHALLENGE_PHASE_CODENAME', 'dev_track_3')
    if not phase_codename:
        raise ValueError('environment variable CHALLENGE_PHASE_CODENAME not defined')
    track = int(phase_codename.split("_")[2])
    phase_codename = phase_codename.split("_")[0]
    if phase_codename == 'test':
        ARGUMENTS.show_to_participant = False

    try:
        challenge_evaluator = ChallengeEvaluator(ARGUMENTS)
        challenge_evaluator.run(ARGUMENTS)
    except Exception as e:
        traceback.print_exc()
        if challenge_evaluator:
            challenge_evaluator.report_challenge_statistics(ARGUMENTS.filename, ARGUMENTS.show_to_participant)
    finally:
        del challenge_evaluator
