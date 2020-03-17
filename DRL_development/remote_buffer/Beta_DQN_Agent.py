"""
    A agent using beta-pessimistic DQN.

    Using CARLA AD challenge Track4 setting.

"""

import carla
from srunner.scenariomanager.timer import GameTime
from srunner.challenge.envs.sensor_interface import SensorInterface
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from srunner.challenge.utils.route_manipulation import downsample_route
# from scenario_runner.srunner.challenge.autoagents.autonomous_agent import AutonomousAgent, Track
from srunner.challenge.autoagents.autonomous_agent import AutonomousAgent, Track
# from PythonAPI.examples.Track4Controller import VehiclePIDController

from collections import deque
import math
import scipy.misc
import numpy as np
import torch
import carla
import matplotlib.pyplot as plt
import copy
import datetime
from enum import Enum
from itertools import product
from collections import namedtuple
import heapq



from srunner.challenge.envs.sensor_interface import SensorInterface
from srunner.scenariomanager.timer import GameTime
from srunner.challenge.utils.route_manipulation import downsample_route
from srunner.challenge.autoagents.autonomous_agent import AutonomousAgent, Track
from DRL_development import util

# check if gpu is avaliable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def distance_to_vehicle(vehicle, target_location):
    """
    A quick fuction to calculate distance between a location and a vehicle
    :param vehicle: selected vehicle(class vehicle)
    :param target_location: desired location(class location)
    :return: distance to selected vehicle
    """
    # v_x = vehicle.get_location().x
    # v_y = vehicle.get_location().y
    # v_z = vehicle.get_location().z
    #
    dx = target_location.x - vehicle.get_location().x
    dy = target_location.y - vehicle.get_location().y
    # dz = target_location.z - vehicle.get_location().z

    distance_to_vehicle = math.sqrt(dx * dx + dy * dy)

    return distance_to_vehicle

class Beta_DQN_Agent(AutonomousAgent):
    """
        Tihs is a beta-pessimistic DQN agent for Track4 CARLA challenge.

    """
    def __init__(self, path_to_conf_file):

        #  RL agent net
        self.algorithm = None
        #  current global plans to reach a destination
        self._global_plan = None

        # this data structure will contain all sensor data
        self.sensor_interface = SensorInterface()

        # generate action space or using args
        self.set_action_space()

        self.state_shape = (3, 300, 200)

        # set ego vehicle
        self.ego_vehicle = None

        # initialize RL element
        self.state = None
        self.next_state = None
        self.reward = None

        # initialize DRL controller
        # todo: get module using module path and name
        # self.controller = DQN_controller()




        # agent's initialization
        self.setup(path_to_conf_file)
        self.generate_action_space()

    def setup_ego_vehicle(self, vehicle):
        """
            API to pass ego vehicle from training env to agent.
        """
        self.ego_vehicle = vehicle


    def set_action_space(self, lon_dim=3, lat_dim=9):
        """
            Generate action space of specified dimension
        """
        self.lon_action_space = self.discrete_action_space(action_dim=3)
        self.lat_action_space = self.discrete_action_space(action_dim=7)

    def discrete_action_space(self, action_dim=7):
        """
            Generate a discrete action space through dimension.
        """
        # if using odd number
        if (action_dim % 2) == 1:
            action_dim -= 1

        action_division = 2 / action_dim

        negative_action = [-1 + i * action_division for i in range(int(action_dim / 2))]
        positive_action = [0 + (1 + i) * action_division for i in range(int(action_dim / 2))]

        action_space = negative_action
        action_space.extend([0])
        action_space.extend(positive_action)

        return action_space

    def setup(self, path_to_conf_file):
        """
        Initialize everything needed by your agent and set the track attribute to the right type:
            Track.ALL_SENSORS : LIDAR, cameras, GPS and speed sensor allowed
            Track.CAMERAS : Only cameras and GPS allowed
            Track.ALL_SENSORS_HDMAP_WAYPOINTS : All sensors and HD Map and waypoints allowed
            Track.SCENE_LAYOUT : No sensors allowed, the agent receives a high-level representation of the scene.
        """
        self.track = Track.CAMERAS

    def sensors(self):
        """
        Define the sensor suite required by the agent

        """
        sensors = [
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
             'width': 300, 'height': 200, 'fov': 100, 'id': 'Mid'},
        ]

        return sensors

    # 根据steer_num和acc_num划分离散动作空间
    def generate_action_space(self):
        for item in product(self.steer_space, self.acc_space):
            self.action_space.append(item)
        self.action_shape = len(self.action_space)

    def run_step(self, state):
        """
        Execute one step of navigation.
        :return: control
        """
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 0.0
        control.hand_brake = False
        state = state.reshape(-1, *self.state_shape)

        # waypoint buffer
        



        # todo: has this part been tested?
        # find 2 nearest waypoint from near_waypoint_buffer
        distance_list = []
        for i, (waypoint, _) in self.near_waypoint_buffer:
            # distance list
            distance = distance_to_vehicle(self.ego_vehicle, waypoint.location)
            add_dict = {'index': i, 'waypoint': waypoint, 'distance': distance}
            distance_list.append(add_dict)

        # not using distance checking
        # if distance_to_vehicle(self.ego_vehicle, waypoint.location) >= min_distance:
        #     pass

        # find min 2 using heapq
        [next_waypoint, last_waypoint] = heapq.nsmallest(2, distance_list, key=lambda s: s['distance'])
        if next_waypoint['index'] >= last_waypoint['index']:
            cup = next_waypoint
            next_waypoint = last_waypoint
            last_waypoint = cup

        # args: vehicle_location
        offset = self.calculate_distance(self.ego_vehicle.get_location(), last_waypoint, next_waypoint)

        # get angle diversion
        angle_diversion =  self.calculete_angle_diversion(self.ego_vehicle.get_location(), last_waypoint, next_waypoint)

        # speed reward
        self.






        state_tensor = torch.tensor(state)
        state_tensor.to(device)

        action_index = self.algorithm.select_action(state_tensor)
        self.action = action_index
        control.steer = self.action_space[action_index][0]
        acc = self.action_space[action_index][1]

        print('acc:', acc)
        # if acc >= 0.0:
        #     control.throttle = acc
        #     control.brake = 0.0
        # else:
        #     control.throttle = 0.0
        #     control.brake = abs(acc)
        control.throttle = acc
        print('throttle:', control.throttle)
        print('steer:', control.steer)

        return control

    # package this method to util
    @staticmethod 
    def calculate_distance(vehicle_location, last_waypoint, next_waypoint):
        """
            todo: check the input type
            args waypoint as carla.location type

        """
        E = np.array([vehicle_location.x, vehicle_location.y])
        A = np.array([last_waypoint.x, last_waypoint.y])
        B = np.array([next_waypoint.x, next_waypoint.y])

        Vector_AE = E - A
        Vector_AB = B - A

        temp = Vector_AE.dot(Vector_AB) / Vector_AB.dot(Vector_AB)
        temp = temp * Vector_AB

        distance = np.linalg.norm(Vector_AE - temp)

        return distance

    @staticmethod
    def calculete_angle_diversion(vehicle_location, last_waypoint, next_waypoint):
        """
            todo: merge 2 functions
        """
        E = np.array([vehicle_location.x, vehicle_location.y])
        A = np.array([last_waypoint.x, last_waypoint.y])
        B = np.array([next_waypoint.x, next_waypoint.y])

        Vector_AE = E - A
        Vector_AB = B - A

        cos_angle = Vector_AB.dot(Vector_AE)/np.linalg.norm(Vector_AE)/np.linalg.norm(Vector_AB)
        abs_cos_angle = abs(cos_angle)

        if abs_cos_angle > 1:
            print("angle_diversion error! check code!")
            abs_cos_angle = np.clip(abs_cos_angle, -1, 1)

        abs_angle = np.arccos(abs_cos_angle)

        # todo: detremine the direction
        return abs_angle




    # set algorithm module as a attribute
    def set_algorithm(self, algorithm):
        self.algorithm = algorithm

    def destroy(self):
        """
        Destroy (clean-up) the agent
        :return:
        """
        pass

    def __call__(self):
        timestamp = GameTime.get_time()
        wallclock = GameTime.get_wallclocktime()
        print('======[Agent] Wallclock_time = {} / Sim_time = {}'.format(wallclock, timestamp))

        control = self.run_step(self.state)
        control.manual_gear_shift = False

        return control

    def all_sensors_ready(self):
        return self.sensor_interface.all_sensors_ready()

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):

        if self.track == Track.CAMERAS or self.track == Track.ALL_SENSORS:
            ds_ids = downsample_route(global_plan_world_coord, 32)

            self._global_plan_world_coord = [(global_plan_world_coord[x][0], global_plan_world_coord[x][1])
                                             for x in ds_ids]
            self._global_plan = [global_plan_gps[x] for x in ds_ids]

        else:  # No downsampling is performed

            self._global_plan = global_plan_gps
            self._global_plan_world_coord = global_plan_world_coord

    def get_state_shape(self):
        return self.state_shape

    def get_action_shape(self):
        return self.action_shape

    # original get reward
    def get_reward(self, reward):
        self.reward = reward




    
    # get reward
    def get_reward(self):
        """
            Get reward according to current status
        """
        lat_offset
        angle

        reached_waypoint =

        collision =
        off_lane =
        off_road =


    # get 2 nearest waypoint
    def get_nearest_waypoint(self):
        """
            Get 2 nearest waypoints around vehicle.
            :return:
        """
        self.near_waypoint_buffer =


    #
    def get_reward(self):
        """
            Get reward.
        :return:
        """
        # off-lane

        # off-

        # reach waypoint


        # todo: move coefficient ndarray to init
        self.coefficient = []
        reward = self.coefficient

    # modified 
    def get_state(self):
        sensor_data = self.sensor_interface.get_data()
        # process image
        image_input = sensor_data['Mid']
        bgr_image = image_input[1][:, :, :3]
        bgr_image = bgr_image.astype(np.float32)
        bgr_image = np.multiply(bgr_image, 1.0 / 255.0)
        bgr_image = np.transpose(bgr_image, (2, 1, 0))
        print('rgb shape:', bgr_image.shape)
        # bgr_image = bgr_image.reshape((1,*self.state_shape))
        print('rgb reshape:', bgr_image.shape)

        if self.state is None:
            print('state is none')
            self.state = bgr_image
        else:
            print('store transition')
            self.next_state = bgr_image
            transition = Transition(self.state, self.action, self.reward, self.next_state)
            self.algorithm.store_transition(transition)
            self.state = self.next_state

        # ==================================================
        # how to get the 
        
        # set a ref speed for normalization
        # km/h
        self.V_ref = 10

        local_coord_frame = 

        # get 2 nearest waypoints

        current_waypoint = 

        next_waypoint = 

        # ego vehicle velocity under local coordinate frame
        V_lat = 
        V_lon = 

        # calculate latral offset respect to route
        lat_offset = 
        lat_diversion_angle
        lat_
        
        

