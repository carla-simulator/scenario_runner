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


import carla
import srunner.challenge.utils.route_configuration_parser as parser


from srunner.scenariomanager.carla_data_provider import CarlaActorPool, CarlaDataProvider


# We import the challenge evaluator here
from srunner.challenge.challenge_evaluator_routes import ChallengeEvaluator


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
    # instance a challenge
    challenge = ChallengeEvaluator(args)

    # retrieve worlds annotations
    world_annotations = parser.parse_annotations_file(args.scenarios)
    # retrieve routes

    routes = []
    route_descriptions_list = parser.parse_routes_file(args.routes)
    for route_description in route_descriptions_list:
        if route_description['town_name'] == args.debug_town:
            routes.append(route_description)
    # find and filter potential scenarios for each of the evaluated routes
    # For each of the routes and corresponding possible scenarios to be evaluated.
    list_scenarios_town = []
    if args.debug_town in world_annotations.keys():
        list_scenarios_town = world_annotations[args.debug_town]

    scenarios_current_type = []
    for scenario in list_scenarios_town:
        if args.debug_scenario == scenario['scenario_type']:
            scenarios_current_type = scenario
            break

    for scenario_configuration in scenarios_current_type['available_event_configurations']:
        scenario_conf = create_configuration_scenario(scenario_configuration, args.debug_scenario)

        client = carla.Client(args.host, int(args.port))
        client.set_timeout(challenge.client_timeout)

        # load the challenge.world variable to be used during the route
        challenge.load_world(client, args.debug_town)
        # Set the actor pool so the scenarios can prepare themselves when needed
        CarlaActorPool.set_world(challenge.world)
        # Also se the Data provider pool.
        CarlaDataProvider.set_world(challenge.world)
        # tick world so we can start.

        challenge.world.tick()
        # create agent
        challenge.agent_instance = getattr(challenge.module_agent, challenge.module_agent.__name__)(args.config)
        correct_sensors, error_message = challenge.valid_sensors_configuration(
            challenge.agent_instance, challenge.track)
        if not correct_sensors:
            # the sensor configuration is illegal
            challenge.report_fatal_error(args.filename, args.show_to_participant, error_message)
            return

        waypoint = routes[0]['trajectory'][0]

        location = carla.Location(x=float(waypoint.attrib['x']),
                                  y=float(waypoint.attrib['y']),
                                  z=float(waypoint.attrib['z']))

        rotation = carla.Rotation(pitch=float(waypoint.attrib['pitch']),
                                  yaw=float(waypoint.attrib['yaw']))

        challenge.prepare_ego_car(carla.Transform(location, rotation))
        CarlaDataProvider.register_actor(challenge.ego_vehicle)

        list_scenarios = []
        list_scenarios += challenge.build_scenario_instances([scenario_conf], args.debug_town)

        # Tick once to start the scenarios.
        print(" Running these scenarios  --- ", list_scenarios)
        for scenario in list_scenarios:
            scenario.scenario.scenario_tree.tick_once()

        challenge.run_route(list_scenarios, None, True)

        # clean up
        for scenario in list_scenarios:
            del scenario
        challenge.cleanup(ego=True)
        challenge.agent_instance.destroy()
        break

    # final measurements from the challenge
    challenge.report_challenge_statistics(args.filename, args.show_to_participant)
    del challenge


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
    PARSER.add_argument('--debug-town', type=str, help='Town used for test', default='Town01')
    PARSER.add_argument('--debug-scenario', type=str, help='Scenario used for test', default='Scenario1')

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
        test_routes(ARGUMENTS)
    finally:
        print("============ OK")
