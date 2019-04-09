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
import sys
import os
from pprint import pprint
import logging

import carla
import srunner.challenge.utils.route_configuration_parser as parser
from srunner.challenge.utils.route_manipulation import interpolate_trajectory

from srunner.scenariomanager.carla_data_provider import CarlaActorPool, CarlaDataProvider


# We import the challenge evaluator here
from srunner.challenge.challenge_evaluator_routes import ChallengeEvaluator


def convert_transform_to_location(transform_vec):

    location_vec = []
    for transform_tuple in transform_vec:
        location_vec.append((transform_tuple[0].location, transform_tuple[1]))

    return location_vec


def create_configuration_scenario(scenario_desc, scenario_type):
    waypoint = scenario_desc['transform']
    parser.convert_waypoint_float(waypoint)

    if 'other_actors' in scenario_desc:
        other_vehicles = scenario_desc['other_actors']
    else:
        other_vehicles = None

    scenario_description = {
        'name': scenario_type,
        'other_actors': other_vehicles,
        'trigger_position': waypoint
    }

    return scenario_description


def test_routes(args):
    """
    Run all routes according to provided commandline args
    """

    # Routes are always visible
    args.route_visible = True
    challenge = ChallengeEvaluator(args)
    # retrieve worlds annotations
    world_annotations = parser.parse_annotations_file(args.scenarios)
    # retrieve routes
    route_descriptions_list = parser.parse_routes_file(args.routes)
    # find and filter potential scenarios for each of the evaluated routes
    # For each of the routes and corresponding possible scenarios to be evaluated.

    route_description = route_descriptions_list[args.route_id]
    # setup world and client assuming that the CARLA server is up and running
    client = carla.Client(args.host, int(args.port))
    client.set_timeout(challenge.client_timeout)

    # load the challenge.world variable to be used during the route
    challenge.load_world(client, route_description['town_name'])
    # Set the actor pool so the scenarios can prepare themselves when needed
    CarlaActorPool.set_client(client)
    CarlaActorPool.set_world(challenge.world)
    # Also se the Data provider pool.
    CarlaDataProvider.set_world(challenge.world)
    # tick world so we can start.
    challenge.world.tick()
    # prepare route's trajectory
    gps_route, route_description['trajectory'] = interpolate_trajectory(challenge.world,
                                                                        route_description['trajectory'])

    CarlaDataProvider.set_ego_vehicle_route(convert_transform_to_location(route_description['trajectory']))

    potential_scenarios_definitions, existent_triggers = parser.scan_route_for_scenarios(route_description,
                                                                                         world_annotations)
    potential_scenarios_definitions = challenge.filter_scenarios(potential_scenarios_definitions,
                                                                 args.removed_scenarios)
    print ("WE HAVE This number of posibilities : ", len(potential_scenarios_definitions))

    # Sample the scenarios to be used for this route instance.
    sampled_scenarios_definitions = challenge.scenario_sampling(potential_scenarios_definitions)
    # create agent
    challenge.agent_instance = getattr(challenge.module_agent, challenge.module_agent.__name__)(args.config)
    correct_sensors, error_message = challenge.valid_sensors_configuration(challenge.agent_instance, challenge.track)
    if not correct_sensors:
        # the sensor configuration is illegal
        challenge.report_fatal_error(args.filename, args.show_to_participant, error_message)
        return

    challenge.agent_instance.set_global_plan(gps_route, route_description['trajectory'])

    # prepare the ego car to run the route.
    # It starts on the first wp of the route

    elevate_transform = route_description['trajectory'][0][0]
    elevate_transform.location.z += 0.5

    challenge.prepare_ego_car(elevate_transform)

    # build the master scenario based on the route and the target.
    challenge.master_scenario = challenge.build_master_scenario(route_description['trajectory'],
                                                                route_description['town_name'])
    list_scenarios = [challenge.master_scenario]

    if args.background:
        background_scenario = challenge.build_background_scenario(route_description['town_name'])
        list_scenarios.append(background_scenario)
    # build the instance based on the parsed definitions.
    print("Definition of the scenarios present on the route ")
    pprint(sampled_scenarios_definitions)
    list_scenarios += challenge.build_scenario_instances(sampled_scenarios_definitions,
                                                         route_description['town_name'])
    if args.debug > 0:
        for scenario in sampled_scenarios_definitions:
            loc = carla.Location(scenario['trigger_position']['x'],
                                 scenario['trigger_position']['y'],
                                 scenario['trigger_position']['z']) + carla.Location(z=2.0)
            challenge.world.debug.draw_point(loc, size=1.0, color=carla.Color(255, 0, 0), life_time=100000)
            challenge.world.debug.draw_string(loc, scenario['name'], draw_shadow=False,
                                              color=carla.Color(0, 0, 255), life_time=100000, persistent_lines=True)

    # Tick once to start the scenarios.
    print(" Running these scenarios  --- ", list_scenarios)
    for scenario in list_scenarios:
        scenario.scenario.scenario_tree.tick_once()

    challenge.run_route(list_scenarios, route_description['trajectory'])

    # statistics recording
    challenge.record_route_statistics(route_description['id'])

    # clean up
    for scenario in list_scenarios:
        scenario.remove_all_actors()
    challenge.cleanup(ego=True)
    challenge.agent_instance.destroy()


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
    PARSER.add_argument('--debug-town', type=str, help='Town used for test', default='Town01')
    PARSER.add_argument('--debug-scenario', type=str, help='Scenario used for test', default='Scenario1')
    PARSER.add_argument('--route-visible', dest='route_visible',
                        action="store_true", help='Run with a visible route')
    PARSER.add_argument('--background', dest='background',
                        action="store_true", help='Add the background scenario around')
    PARSER.add_argument('--show-to-participant', type=bool, help='Show results to participant?', default=True)
    PARSER.add_argument('--routes',
                        help='Name of the route to be executed. Point to the route_xml_file to be executed.')
    PARSER.add_argument('--remove', dest='removed_scenarios',  nargs='+',  default=[],
                        help='Scenarios to remove')
    PARSER.add_argument('--scenarios',
                        help='Name of the scenario annotation file to be mixed with the route.')

    PARSER.add_argument("-id", "--route_id", type=int, default=0, help="the id of the route you want to test")
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
        test_routes(ARGUMENTS)
    finally:
        print("============ OK")
