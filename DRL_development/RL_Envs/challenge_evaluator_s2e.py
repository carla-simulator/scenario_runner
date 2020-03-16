#!/usr/bin/env python
# authors: Guo Youtian
#
# This work is modified challenge evaluator_routes
# s2e means the route is specified by starting point and ending point
# The 1.0 Version do not contain scenario module

from __future__ import print_function
import argparse
from argparse import RawTextHelpFormatter
import sys
import os
import traceback
import datetime
import time
import importlib
import math
import carla

import py_trees

from agents.navigation.local_planner import RoadOption
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO

import srunner.challenge.utils.route_configuration_parser as parser
from srunner.challenge.autoagents.autonomous_agent import Track
from srunner.scenariomanager.timer import GameTime
from srunner.challenge.envs.scene_layout_sensors import SceneLayoutReader, ObjectFinder
from srunner.challenge.envs.sensor_interface import CallBack, CANBusSensor, HDMapReader
from srunner.scenariomanager.carla_data_provider import CarlaActorPool, CarlaDataProvider
from srunner.tools.config_parser import ActorConfiguration, ScenarioConfiguration, ActorConfigurationData
from srunner.challenge.utils.route_manipulation import interpolate_trajectory, clean_route
from srunner.scenarios.master_scenario import MasterScenario
from srunner.scenariomanager.traffic_events import TrafficEventType


from srunner.challenge.autoagents.RLagent import RLAgent
from srunner.challenge.algorithm.dqn import DQNAlgorithm

PENALTY_COLLISION_STATIC = 6
PENALTY_COLLISION_VEHICLE = 6
PENALTY_COLLISION_PEDESTRIAN = 9
PENALTY_TRAFFIC_LIGHT = 3
PENALTY_WRONG_WAY = 2
PENALTY_SIDEWALK_INVASION = 2
PENALTY_STOP = 2

EPISODES = 100

def convert_transform_to_location(transform_vec):

    location_vec = []
    for transform_tuple in transform_vec:
        location_vec.append((transform_tuple[0].location, transform_tuple[1]))

    return location_vec


class Modified_ChallengeEvaluator(object):
    """
    Provisional code to evaluate AutonomousAgent performance
    """
    MAX_ALLOWED_RADIUS_SENSOR = 5.0
    SECONDS_GIVEN_PER_METERS = 0.4
    MAX_CONNECTION_ATTEMPTS = 5

    def __init__(self,args):

        # retrieving scenario_runner root
        scenario_runner_root = os.getenv('ROOT_SCENARIO_RUNNER', '/workspace/scenario_runner')

        # remaining simulation time available for this time in seconds
        challenge_time_available = int(os.getenv('CHALLENGE_TIME_AVAILABLE', '1080000'))
        self.challenge_time_available = challenge_time_available

        if args.spectator:
            self.spectator = args.spectator
        else:
            self.spectator = False

        self.track = args.track

        self.debug = args.debug
        self.ego_vehicle = None
        self._system_error = False
        self.actors = []

        # Tunable parameters
        self.client_timeout = 30.0  # in seconds
        self.wait_for_world = 20.0  # in seconds

        # CARLA world and scenario handlers
        self.world = None
        self.agent_instance = None

        self.master_scenario = None
        # self.background_scenario = None
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

        self.map = args.map

        self.config = args.config

        self.starting_point = args.starting
        self.ending_point = args.ending

        self.rendering = args.rendering

        # setup world and client assuming that the CARLA server is up and running
        self.client = carla.Client(args.host, int(args.port))
        self.client.set_timeout(self.client_timeout)


        # For debugging
        self.route_visible = self.debug > 0

        # Try to load the world and start recording
        # If not successful stop recording and continue with next iteration

        self.load_world(self.client, self.map)

        # route calculate
        self.gps_route, self.route = self.calculate_route()
        self.route_timeout = self.estimate_route_timeout()

        # agent algorithm
        self.agent_algorithm = None



    def reach_ending_point(self):
        return False

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
            self.ego_vehicle = CarlaActorPool.request_new_actor('vehicle.lincoln.mkz2017', start_transform, hero=True)
            # setup sensors
            if self.agent_instance is not None:
                self.setup_sensors(self.agent_instance.sensors(), self.ego_vehicle)
            self.ego_vehicle.set_transform(start_transform)

    def calculate_route(self):
        """
        This function calculate a route for giving starting_point and ending_point
        :return: route (includeing Waypoint.transform & RoadOption)
        """
        starting_location = carla.Location(x=float(self.starting_point.split("_")[0]),
                                        y=float(self.starting_point.split("_")[1]),
                                        z=float(self.starting_point.split("_")[2]))

        ending_location = carla.Location(x=float(self.ending_point.split("_")[0]),
                                        y=float(self.ending_point.split("_")[1]),
                                        z=float(self.ending_point.split("_")[2]))

        # returns list of (carla.Waypoint.transform, RoadOption) from origin to destination
        coarse_route = []
        coarse_route.append(starting_location)
        coarse_route.append(ending_location)

        return  interpolate_trajectory(self.world, coarse_route)

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
            self.world.wait_for_tick(self.wait_for_world)
    
    def load_world(self, client, town_name):

        self.world = client.load_world(town_name)
        self.timestamp = self.world.wait_for_tick(self.wait_for_world)
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        if self.rendering == False:
            settings.no_rendering_mode = True
        self.world.apply_settings(settings)

    def build_master_scenario(self, route, town_name, timeout=300):
        # We have to find the target.
        # we also have to convert the route to the expected format
        master_scenario_configuration = ScenarioConfiguration()
        master_scenario_configuration.target = route[-1][0]  # Take the last point and add as target.
        master_scenario_configuration.route = convert_transform_to_location(route)
        master_scenario_configuration.town = town_name
        # TODO THIS NAME IS BIT WEIRD SINCE THE EGO VEHICLE  IS ALREADY THERE, IT IS MORE ABOUT THE TRANSFORM
        master_scenario_configuration.ego_vehicle = ActorConfigurationData('vehicle.lincoln.mkz2017',self.ego_vehicle.get_transform())
        master_scenario_configuration.trigger_point = self.ego_vehicle.get_transform()
        CarlaDataProvider.register_actor(self.ego_vehicle)

        # Provide an initial blackboard entry to ensure all scenarios are running correctly
        blackboard = py_trees.blackboard.Blackboard()
        blackboard.set('master_scenario_command', 'scenarios_running')

        return MasterScenario(self.world, self.ego_vehicle, master_scenario_configuration,timeout=timeout, debug_mode=self.debug > 1)

    def estimate_route_timeout(self):
        route_length = 0.0  # in meters

        prev_point = self.route[0][0]
        for current_point, _ in self.route[1:]:
            dist = current_point.location.distance(prev_point.location)
            route_length += dist
            prev_point = current_point

        return int(self.SECONDS_GIVEN_PER_METERS * route_length)

    def route_is_running(self):
        """
            Test if the route is still running.
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

    def worldReset(self):
        """
        Reset the world
        """
        # prepare the ego car to run the route.
        # It starts on the first wp of the route
        self.list_scenarios = []
        self.cleanup(True)

        # Set the actor pool so the scenarios can prepare themselves when needed
        CarlaActorPool.set_client(self.client)
        CarlaActorPool.set_world(self.world)
        # Also se the Data provider pool.
        CarlaDataProvider.set_world(self.world)
        CarlaDataProvider.set_ego_vehicle_route(convert_transform_to_location(self.route))

        # # create agent
        # self.agent_instance = getattr(self.module_agent, self.module_agent.__name__)(self.config)
        # self.agent_instance.set_global_plan(self.gps_route, self.route)

        # creat RLagent
        self.agent_instance = RLAgent(self.config)
        self.agent_instance.set_global_plan(self.gps_route, self.route)
        
        if self.agent_algorithm is None:
            self.agent_algorithm = DQNAlgorithm(self.agent_instance.get_state_shape(), self.agent_instance.get_action_shape())
        self.agent_instance.set_algorithm(self.agent_algorithm)

        elevate_transform = self.route[0][0]
        elevate_transform.location.z += 0.5
        self.prepare_ego_car(elevate_transform)

        GameTime.restart()


    def run_route(self, trajectory, no_master=False):
        print('route is running',self.route_is_running())
        while no_master or self.route_is_running():
            # update all scenarios
            GameTime.on_carla_tick(self.timestamp)
            CarlaDataProvider.on_carla_tick()

            # update all scenarios
            # read state and take action
            # 调用callback callback读取传感器状态并采取动作，保存状态功能写在agent中
            self.agent_instance.get_state()
            ego_action = self.agent_instance()

            for scenario in self.list_scenarios:
                scenario.scenario.scenario_tree.tick_once()
            
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
            # 打分
            # 测试非训练框架时屏蔽此句
            total_score, route_score, infractions_score = self.compute_current_statistics()

            # send the reward to agent
            # 把reward传给agent保存
            # 测试非训练框架时屏蔽此句
            self.agent_instance.get_reward(total_score)

            #learn
            # 更新Q网络
            # 测试非训练框架时屏蔽此句
            self.agent_instance.algorithm.update()


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
                    self.timestamp = self.world.wait_for_tick(self.wait_for_world)
                    break
                except Exception:
                    attempts += 1
                    print('======[WARNING] The server is frozen [{}/{} attempts]!!'.format(attempts,
                                                                                            self.MAX_CONNECTION_ATTEMPTS))
                    time.sleep(2.0)
                    continue
        

    def valid_sensors_configuration(self, agent, track):

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

    def load_environment_and_run(self, args):

    
        #correct_sensors, error_message = self.valid_sensors_configuration(self.agent_instance, self.track)

        #if not correct_sensors:
            # the sensor configuration is illegal
        #    sys.exit(-1)

                # build the master scenario based on the route and the target.


        # train for EPISODES times
        for i in range(EPISODES):
            self.worldReset()
            self.master_scenario = self.build_master_scenario(self.route,self.map,timeout=self.route_timeout)
            self.list_scenarios.append(self.master_scenario)

            # main loop!
            self.run_route(self.route)
            self.world.tick()
  

    def run(self, args):
        """
        Run route according to provided commandline args
        """

        # tick world so we can start.
        self.world.tick()


        # Try to run the route
        # If something goes wrong, still take the current score, and continue
        try:
            self.load_environment_and_run(args)
        except Exception as e:
            if self.debug > 0:
                traceback.print_exc()
                raise
            if self._system_error or not self.agent_instance:
                print(e)
                sys.exit(-1)


        # clean up
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)
        self.agent_instance.destroy()
        self.agent_instance = None
        self.cleanup(ego=True)






if __name__ == '__main__':
    DESCRIPTION = ("CARLA AD Challenge evaluation: evaluate your Agent in CARLA scenarios\n")

    PARSER = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=RawTextHelpFormatter)
    PARSER.add_argument('--host', default='localhost',
                        help='IP of the host server (default: localhost)')
    PARSER.add_argument('--port', default='2000', help='TCP port to listen to (default: 2000)')
    PARSER.add_argument("-a", "--agent", type=str, help="Path to Agent's py file to evaluate")
    PARSER.add_argument("--config", type=str, help="Path to Agent's configuration file", default=" ")
    PARSER.add_argument("-m", "--map", type=str, help="Town name")
    PARSER.add_argument("-s", "--starting", type=str, help="Agent's Starting point(Transform format:(x,y,z,pitch,yaw,row))")
    PARSER.add_argument("-e", "--ending", type=str, help="Agent's Ending point(Transform format:(x,y,z,pitch,yaw,row))")
    PARSER.add_argument('--track', type=int, help='track type', default=4)
    PARSER.add_argument('--spectator', type=bool, help='Switch spectator view on?', default=True)
    PARSER.add_argument('--rendering', type=bool, help='Switch rendering on?', default=True)
    PARSER.add_argument('--debug', type=int, help='Run with debug output', default=0)
 
    ARGUMENTS = PARSER.parse_args()

    CARLA_ROOT = os.environ.get('CARLA_ROOT')
    ROOT_SCENARIO_RUNNER = os.environ.get('ROOT_SCENARIO_RUNNER')

    if not CARLA_ROOT:
        print("Error. CARLA_ROOT not found. Please run setup_environment.sh first.")
        sys.exit(0)

    if not ROOT_SCENARIO_RUNNER:
        print("Error. ROOT_SCENARIO_RUNNER not found. Please run setup_environment.sh first.")
        sys.exit(0)

    if ARGUMENTS.map is None:
        print("Please specify a map \n\n")
        PARSER.print_help(sys.stdout)
        sys.exit(0)

    if ARGUMENTS.starting is None:
        print("Please specify a strating point(Transform)  \n\n")
        PARSER.print_help(sys.stdout)
        sys.exit(0)

    
    if ARGUMENTS.ending is None:
        print("Please specify a ending point(Transform)  \n\n")
        PARSER.print_help(sys.stdout)
        sys.exit(0)
    
    ARGUMENTS.carla_root = CARLA_ROOT
    challenge_evaluator = None

    try:
        challenge_evaluator = Modified_ChallengeEvaluator(ARGUMENTS)
        challenge_evaluator.run(ARGUMENTS)
    except Exception as e:
        traceback.print_exc()
        if challenge_evaluator:
            challenge_evaluator.report_challenge_statistics(ARGUMENTS.filename, ARGUMENTS.show_to_participant)
    finally:
        del challenge_evaluator