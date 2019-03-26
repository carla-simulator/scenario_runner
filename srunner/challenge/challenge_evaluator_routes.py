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
import importlib
import sys
import os
import json
import random
import py_trees

import xml.etree.ElementTree as ET

import carla
import srunner.challenge.utils.route_configuration_parser as parser
from srunner.challenge.envs.scene_layout_sensors import SceneLayoutReader, ObjectFinder
from srunner.challenge.envs.sensor_interface import CallBack, CANBusSensor, HDMapReader
from srunner.challenge.autoagents.autonomous_agent import Track


from srunner.scenariomanager.carla_data_provider import CarlaActorPool, CarlaDataProvider

from srunner.scenarios.control_loss import ControlLoss
from srunner.scenarios.follow_leading_vehicle import FollowLeadingVehicle
from srunner.scenarios.object_crash_vehicle import DynamicObjectCrossing
from srunner.scenarios.object_crash_intersection import VehicleTurningRight, VehicleTurningLeft
from srunner.scenarios.opposite_vehicle_taking_priority import OppositeVehicleRunningRedLight
from srunner.scenarios.signalized_junction_left_turn import SignalizedJunctionLeftTurn
from srunner.scenarios.signalized_junction_right_turn import SignalizedJunctionRightTurn
from srunner.scenarios.no_signal_junction_crossing import NoSignalJunctionCrossing
from srunner.scenarios.maneuver_opposite_direction import ManeuverOppositeDirection
from srunner.scenarios.master_scenario import MasterScenario

# The configuration parser

from srunner.scenarios.config_parser import ActorConfiguration, ScenarioConfiguration, \
    RouteConfiguration, ActorConfigurationData
from srunner.scenariomanager.traffic_events import TrafficEvent, TrafficEventType

from srunner.challenge.utils.route_manipulation import interpolate_trajectory


number_class_translation = {

    "Scenario1": [ControlLoss],
    "Scenario2": [FollowLeadingVehicle],   # TODO there is more than one class depending on the scenario configuration
    "Scenario3": [DynamicObjectCrossing],
    "Scenario4": [VehicleTurningRight, VehicleTurningLeft],
    "Scenario5": [],
    "Scenario6": [ManeuverOppositeDirection],
    "Scenario7": [OppositeVehicleRunningRedLight],
    "Scenario8": [SignalizedJunctionLeftTurn],
    "Scenario9": [SignalizedJunctionRightTurn],
    "Scenario10": [NoSignalJunctionCrossing]

}
# Util functions


def convert_json_to_actor(actor_dict):
    node = ET.Element('waypoint')
    node.set('x', actor_dict['x'])
    node.set('y', actor_dict['y'])
    node.set('z', actor_dict['z'])
    node.set('yaw', actor_dict['yaw'])

    return ActorConfiguration(node)


def convert_json_to_transform(actor_dict):

    return carla.Transform(location=carla.Location(x=float(actor_dict['x']), y=float(actor_dict['y']),
                                                   z=float(actor_dict['z'])),
                           rotation=carla.Rotation(roll=0.0, pitch=0.0, yaw=float(actor_dict['yaw'])))


class ChallengeEvaluator(object):

    """
    Provisional code to evaluate AutonomousAgent performance
    """

    def __init__(self, args):
        phase_codename = args.split
        self.phase = phase_codename.split("_")[0]
        self.track = int(phase_codename.split("_")[-1])

        self.ego_vehicle = None
        self.actors = []
        self.statistics_routes = []

        # Tunable parameters
        self.client_timeout = 15.0  # in seconds
        self.wait_for_world = 10.0  # in seconds

        # CARLA world and scenario handlers
        self.world = None
        self.agent_instance = None

        self.output_scenario = []
        self.master_scenario = None
        # first we instantiate the Agent
        if args.agent is not None:
            module_name = os.path.basename(args.agent).split('.')[0]
            sys.path.insert(0, os.path.dirname(args.agent))
            self.module_agent = importlib.import_module(module_name)
        self._sensors_list = []
        self._hop_resolution = 2.0

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

    def __del__(self):
        """
        Cleanup and delete actors, ScenarioManager and CARLA world
        """

        self.cleanup(True)
        if self.world is not None:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
            del self.world

    def prepare_ego_car(self, start_transform):
        """
        Spawn or update all scenario actors according to
        a certain start position.
        """

        # If ego_vehicle already exists, just update location
        # Otherwise spawn ego vehicle
        if self.ego_vehicle is None:
            # TODO: the model is now hardcoded but that can change in a future.
            self.ego_vehicle = CarlaActorPool.setup_actor('vehicle.lincoln.mkz2017', start_transform, True)
        else:
            self.ego_vehicle.set_transform(start_transform)

        # setup sensors
        if self.agent_instance is not None:
            self.setup_sensors(self.agent_instance.sensors(), self.ego_vehicle)

    def draw_waypoints(self, waypoints, vertical_shift, persistency=-1):
        """
        Draw a list of waypoints at a certain height given in vertical_shift.
        :param waypoints: list or iterable container with the waypoints to draw
        :param vertical_shift: height in meters
        :return:
        """
        for w in waypoints:
            wp = w[0].transform.location + carla.Location(z=vertical_shift)
            self.world.debug.draw_point(wp, size=0.1, color=carla.Color(0, 255, 0), life_time=persistency)

    def scenario_sampling(self, potential_scenarios_definitions):
        """
        The function used to sample the scenarios that are going to happen for this route.
        :param potential_scenarios_definitions: all the scenarios to be sampled
        :return: return the ones sampled for this case.
        """
        # The idea is to randomly sample a scenario per trigger position.
        sampled_scenarios = []
        for trigger in potential_scenarios_definitions.keys():
            possible_scenarios = potential_scenarios_definitions[trigger]
            sampled_scenarios.append(random.choice(possible_scenarios))

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
            print(" waiting for one data reading from sensors...")
            self.world.tick()
            self.world.wait_for_tick()

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

    def build_master_scenario(self, route, town_name):
        # We have to find the target.
        # we also have to convert the route to the expected format
        master_scenario_configuration = ScenarioConfiguration()
        master_scenario_configuration.target = route[-1][0]  # Take the last point and add as target.
        master_scenario_configuration.route = route
        master_scenario_configuration.town = town_name
        # TODO THIS NAME IS BIT WEIRD SINCE THE EGO VEHICLE  IS ALREADY THERE, IT IS MORE ABOUT THE TRANSFORM
        master_scenario_configuration.ego_vehicle = ActorConfigurationData('vehicle.lincoln.mkz2017',
                                                                           self.ego_vehicle.get_transform())
        master_scenario_configuration.trigger_point = self.ego_vehicle.get_transform()

        return MasterScenario(self.world, self.ego_vehicle, master_scenario_configuration)

    def build_scenario_instances(self, scenario_definition_vec, town_name):
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
            #  TODO for now I dont know how to disambiguate this part.
            ScenarioClass = possibility_vec[0]
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
            scenario_configuration.trigger_point = egoactor_trigger_position
            scenario_configuration.ego_vehicle = ActorConfigurationData('vehicle.lincoln.mkz2017',
                                                                        self.ego_vehicle.get_transform())

            scenario_instance = ScenarioClass(self.world, self.ego_vehicle, scenario_configuration)
            scenario_instance_vec.append(scenario_instance)

        return scenario_instance_vec

    def route_is_running(self):
        """
            The master scenario tests if the route is still running.
        """
        if self.master_scenario is None:
            raise ValueError('You should not run a route without a master scenario')

        return self.master_scenario.scenario.scenario_tree.status == py_trees.common.Status.RUNNING

    def record_route_statistics(self, route_id):
        """
          This function is intended to be called from outside and provide
          statistics about the scenario (human-readable, for the CARLA challenge.)
        """
        PENALTY_COLLISION_STATIC = 10
        PENALTY_COLLISION_VEHICLE = 10
        PENALTY_COLLISION_PEDESTRIAN = 30
        PENALTY_TRAFFIC_LIGHT = 10
        PENALTY_WRONG_WAY = 5

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

            elif event.get_type() == TrafficEventType.ROUTE_COMPLETED:
                score_route = 100.0
                target_reached = True
            elif event.get_type() == TrafficEventType.ROUTE_COMPLETION:
                if not target_reached:
                    if event.get_dict() and 'route_completed' in event.get_dict():
                        score_route = event.get_dict()['route_completed']
                    else:
                        score_route = 0.0

            score_composed = max(score_route - score_penalty, 0)

            return_message += "\n=================================="
            return_message += "\n==[{}] [Score = {:.2f} : (route_score={}, infractions=-{})]".format(result,
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

            if list_wrong_way:
                return_message += "\n===== Wrong way:"
                for item in list_wrong_way:
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
        n_routes = len(self.statistics_routes)
        score_composed = 0.0
        score_route = 0.0
        score_penalty = 0.0
        help_message = ""

        for stats in self.statistics_routes:
            score_composed += stats['score_composed'] / float(n_routes)
            score_route += stats['score_route'] / float(n_routes)
            score_penalty += stats['score_penalty'] / float(n_routes)
            help_message += "{}\n\n".format(stats['help_text'])

        if self.phase == 'validation' or self.phase == 'test':
            help_message = "No metadata available for this phase"

        # create json structure
        json_data = {
            'submission_status': 'FINISHED',
            'stderr': help_message,
            'results': [
                {
                    'split': self.phase,
                    'show_to_participant': show_to_participant,
                    'accuracies': {
                        'avg. route points': score_route,
                        'infraction points': score_penalty,
                        'total avg.': score_composed
                    }
                }],
        }

        with open(filename, "w+") as fd:
            fd.write(json.dumps(json_data, indent=4))

    def report_fatal_error(self, filename, show_to_participant, error_message):

        # create json structure
        json_data = {
            'submission_status': 'FAILED',
            'stderr': error_message,
            'results': [
                {
                    'split': self.phase,
                    'show_to_participant': show_to_participant,
                    'accuracies': {
                        'avg. route points': 0,
                        'infraction points': 0,
                        'total avg.': 0
                    }
                }],
        }

        with open(filename, "w+") as fd:
            fd.write(json.dumps(json_data, indent=4))

    def load_world(self, client, town_name):
        if self.world is not None:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
        self.world = client.load_world(town_name)
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        self.world.apply_settings(settings)

        self.world.wait_for_tick()

    def valid_sensors_configuration(self, agent, track):
        if Track(track) != agent.track:
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
                        'sensor.object_finder') or sensor['type'].startswith(
                        'sensor.other.gnss')):
                    return False, "Illegal sensor used for Track [{}]!".format(agent.track)

        return True, ""

    def run(self, args):
        """
        Run all routes according to provided commandline args
        """

        # retrieve worlds annotations
        world_annotations = parser.parse_annotations_file(args.scenarios)
        # retrieve routes
        route_descriptions_list = parser.parse_routes_file(args.routes)
        # find and filter potential scenarios for each of the evaluated routes
        # For each of the routes and corresponding possible scenarios to be evaluated.

        for route_description in route_descriptions_list:
            # setup world and client assuming that the CARLA server is up and running
            client = carla.Client(args.host, int(args.port))
            client.set_timeout(self.client_timeout)

            # load the self.world variable to be used during the route
            self.load_world(client, route_description['town_name'])
            # Set the actor pool so the scenarios can prepare themselves when needed
            CarlaActorPool.set_world(self.world)
            # Also se the Data provider pool.
            CarlaDataProvider.set_world(self.world)
            # tick world so we can start.
            self.world.tick()
            # prepare route's trajectory
            gps_route, route_description['trajectory'] = interpolate_trajectory(self.world,
                                                                                route_description['trajectory'])

            potential_scenarios_definitions, existent_triggers = parser.scan_route_for_scenarios(route_description,
                                                                                                 world_annotations)
            # Sample the scenarios to be used for this route instance.
            sampled_scenarios_definitions = self.scenario_sampling(potential_scenarios_definitions)
            # create agent
            self.agent_instance = getattr(self.module_agent, self.module_agent.__name__)(args.config)
            correct_sensors, error_message = self.valid_sensors_configuration(self.agent_instance, self.track)
            if not correct_sensors:
                # the sensor configuration is illegal
                self.report_fatal_error(args.filename, args.show_to_participant, error_message)
                return

            self.agent_instance.set_global_plan(gps_route)

            # prepare the ego car to run the route.
            # It starts on the first wp of the route
            self.prepare_ego_car(route_description['trajectory'][0][0].transform)
            # build the master scenario based on the route and the target.
            self.master_scenario = self.build_master_scenario(route_description['trajectory'],
                                                              route_description['town_name'])
            list_scenarios = [self.master_scenario]
            # build the instance based on the parsed definitions.
            list_scenarios += self.build_scenario_instances(sampled_scenarios_definitions,
                                                            route_description['town_name'])

            # Tick once to start the scenarios.
            print (" Running these scenarios  --- ", list_scenarios)
            for scenario in list_scenarios:
                scenario.scenario.scenario_tree.tick_once()

            while self.route_is_running():
                # update all scenarios
                for scenario in list_scenarios:
                    scenario.scenario.scenario_tree.tick_once()
                # ego vehicle acts
                ego_action = self.agent_instance()
                self.ego_vehicle.apply_control(ego_action)

                if args.route_visible:
                    self.draw_waypoints(route_description['trajectory'],
                                        vertical_shift=1.0, persistency=scenario.timeout)

                # time continues
                self.world.tick()

            # statistics recording
            self.record_route_statistics(route_description['id'])

            # clean up
            for scenario in list_scenarios:
                del scenario
            self.cleanup(ego=True)
            self.agent_instance.destroy()
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
    PARSER.add_argument('--debug', action="store_true", help='Run with debug output')
    PARSER.add_argument('--filename', type=str, help='Filename to store challenge results', default='results.json')
    PARSER.add_argument('--split', type=str, help='Challenge split', default='dev_track_2')
    PARSER.add_argument('--route-visible', dest='route_visible',
                        action="store_true", help='Run with a visible route')
    PARSER.add_argument('--show-to-participant', type=bool, help='Show results to participant?', default=True)
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

    if ARGUMENTS.routes is None:
        print("Please specify a path to a route file  '--routes path-to-route'\n\n")
        PARSER.print_help(sys.stdout)
        sys.exit(0)

    if ARGUMENTS.scenarios is None:
        print("Please specify a path to a scenario specification file  '--scenarios path-to-file'\n\n")
        PARSER.print_help(sys.stdout)
        sys.exit(0)

    ARGUMENTS.carla_root = CARLA_ROOT
    challenge_evaluator = None
    try:
        challenge_evaluator = ChallengeEvaluator(ARGUMENTS)
        challenge_evaluator.run(ARGUMENTS)
    finally:
        del challenge_evaluator
