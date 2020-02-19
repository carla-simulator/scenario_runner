import unittest
import os

import srunner.challenge.utils.route_configuration_parser as parser
from srunner.challenge.challenge_evaluator_routes import ChallengeEvaluator, convert_json_to_transform

from srunner.scenariomanager.carla_data_provider import CarlaActorPool

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider


from srunner.challenge.utils.route_manipulation import interpolate_trajectory

import traceback
import carla




def convert_waypoint_float(waypoint):

    waypoint['x'] = float(waypoint['x'])
    waypoint['y'] = float(waypoint['y'])
    waypoint['z'] = float(waypoint['z'])
    waypoint['yaw'] = float(waypoint['yaw'])


class Arguments():

    def __init__(self):
        self.agent = None
        self.use_docker = False
        self.host = '127.0.0.1'
        self.port = 2000
        self.split = 'dev_track_1'
        self.route_visible = False


class TestRouteSpawn(unittest.TestCase):

    def __init__(self, name='runTest'):
        unittest.TestCase.__init__(self, name)
        self.root_route_file_position = 'srunner/challenge/'


    def test_possible_spawns(self):

        args = Arguments()
        client = carla.Client(args.host, int(args.port))
        client.set_timeout(25.0)
        challenge = ChallengeEvaluator(args)

        filename = os.path.join(self.root_route_file_position, 'routes_training.xml')
        list_route_descriptions = parser.parse_routes_file(filename)

        # Which type of file is expected ????

        # For each of the routes to be evaluated.
        for route_description in list_route_descriptions:
            challenge.world = client.load_world(route_description['town_name'])

            # Set the actor pool so the scenarios can prepare themselves when needed
            CarlaActorPool.set_world(challenge.world)

            CarlaDataProvider.set_world(challenge.world)
            # find and filter potential scenarios
            # Returns the iterpolation in a different format

            challenge.world.wait_for_tick()

            gps_route, route_description['trajectory'] = interpolate_trajectory(challenge.world,
                                                                                route_description['trajectory'])

            #print (gps_route)
            #print (route_description['trajectory'])

            elevate_transform = route_description['trajectory'][0][0].transform
            elevate_transform.location.z += 0.5
            #print (elevate_transform)
            challenge.prepare_ego_car(elevate_transform)


            challenge.cleanup(ego=True)

            #print ("Failed Scenarios ", list_failed)
