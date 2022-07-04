#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Scenarios in which another (opposite) vehicle 'illegally' takes
priority, e.g. by running a red traffic light.
"""

from __future__ import print_function

import py_trees
import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import ActorDestroy, SwitchOutsideRouteLanesTest, \
    BasicAgentBehavior, BicycleFlow, ConstantVelocityAgentBehavior
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import DriveDistance
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tests.carla_mocks.agents.navigation.local_planner import RoadOption
from srunner.tools.background_manager import (HandleStartAccidentScenario,
                                              HandleEndAccidentScenario,
                                              LeaveSpaceInFront,
                                              ChangeOppositeBehavior)


class Accident(BasicScenario):
    """
    This class holds everything required for a scenario in which there is an accident
    in front of the ego, forcing it to react. A police vehicle is located before
    two other cars that have been in an accident.
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=180):
        """
        Setup all relevant parameters and create scenario
        and instantiate scenario manager
        """
        self._world = world
        self._map = CarlaDataProvider.get_map()
        self.timeout = timeout
        self._drive_distance = 120
        self._distance_to_accident = 100
        self._offset = 0.75
        self._first_distance = 10
        self._second_distance = 6
        self._accident_wp = None

        self._lights = carla.VehicleLightState.Special1 | carla.VehicleLightState.Special2

        super(Accident, self).__init__("Accident",
                                       ego_vehicles,
                                       config,
                                       world,
                                       randomize,
                                       debug_mode,
                                       criteria_enable=criteria_enable)

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        starting_wp = self._map.get_waypoint(config.trigger_points[0].location)
        accident_wps = starting_wp.next(self._distance_to_accident)
        pre_accident_wps = starting_wp.next(self._distance_to_accident/2)
        if not accident_wps: 
            raise ValueError("Couldn't find a viable position to set up the accident actors")
        if not pre_accident_wps: 
            raise ValueError("Couldn't find a viable position to set up the accident actors")
        self._accident_wp = accident_wps[0]

        # Create the police vehicle
        displacement = self._offset * self._accident_wp.lane_width / 2
        r_vec = self._accident_wp.transform.get_right_vector()
        w_loc = self._accident_wp.transform.location
        w_loc += carla.Location(x=displacement * r_vec.x, y=displacement * r_vec.y)
        police_transform = carla.Transform(w_loc, self._accident_wp.transform.rotation)
        police_car = CarlaDataProvider.request_new_actor('vehicle.dodge.charger_police_2020', police_transform)
        lights = police_car.get_light_state()
        lights |= self._lights
        police_car.set_light_state(carla.VehicleLightState(lights))
        self.other_actors.append(police_car)

        # Create the first vehicle that has been in the accident
        vehicle_wps = self._accident_wp.next(self._first_distance)
        if not vehicle_wps: 
            raise ValueError("Couldn't find a viable position to set up the accident actors")
        vehicle_wp = vehicle_wps[0]
        self._accident_wp = pre_accident_wps[0]
        displacement = self._offset * vehicle_wp.lane_width / 2
        r_vec = vehicle_wp.transform.get_right_vector()
        w_loc = vehicle_wp.transform.location
        w_loc += carla.Location(x=displacement * r_vec.x, y=displacement * r_vec.y)
        vehicle_1_transform = carla.Transform(w_loc, vehicle_wp.transform.rotation)
        vehicle_1_car = CarlaDataProvider.request_new_actor('vehicle.tesla.model3', vehicle_1_transform)
        self.other_actors.append(vehicle_1_car)

        # Create the second vehicle that has been in the accident
        vehicle_wps = vehicle_wp.next(self._second_distance)
        if not vehicle_wps: 
            raise ValueError("Couldn't find a viable position to set up the accident actors")
        vehicle_wp = vehicle_wps[0]

        displacement = self._offset * vehicle_wp.lane_width / 2
        r_vec = vehicle_wp.transform.get_right_vector()
        w_loc = vehicle_wp.transform.location
        w_loc += carla.Location(x=displacement * r_vec.x, y=displacement * r_vec.y)
        vehicle_2_transform = carla.Transform(w_loc, vehicle_wp.transform.rotation)
        vehicle_2_car = CarlaDataProvider.request_new_actor('vehicle.mercedes.coupe_2020', vehicle_2_transform)
        self.other_actors.append(vehicle_2_car)

    def _create_behavior(self):
        """
        The vehicle has to drive the whole predetermined distance.
        """
        root = py_trees.composites.Sequence()
        if self.route_mode:
            root.add_child(HandleStartAccidentScenario(self._accident_wp, self._distance_to_accident))
        root.add_child(DriveDistance(self.ego_vehicles[0], self._drive_distance))
        if self.route_mode:
            root.add_child(HandleEndAccidentScenario())
        root.add_child(ActorDestroy(self.other_actors[0]))
        root.add_child(ActorDestroy(self.other_actors[1]))
        root.add_child(ActorDestroy(self.other_actors[2]))

        return root

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        if self.route_mode:
            return []
        return [CollisionTest(self.ego_vehicles[0])]

    def __del__(self):
        """
        Remove all actors and traffic lights upon deletion
        """
        self.remove_all_actors()
  
class BicycleFlowAtSideLane(BasicScenario):
    """
    Added the dangerous scene of ego vehicles driving on roads without sidewalks,
    with three bicycles encroaching on some roads in front.
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=180):
        """
        Setup all relevant parameters and create scenario
        and instantiate scenario manager
        """
        self._world = world
        self._map = CarlaDataProvider.get_map()
        self.timeout = timeout
        self._drive_distance = 100
        self._offset = [0.6,0.75,0.9]
        self._bicycle_wp = []
        self._target_location=None
        self._plan=[]

        if 'distance' in config.other_parameters:
            self._distance_to_Trigger = [
                float(config.other_parameters['distance']['first']),
                float(config.other_parameters['distance']['second']),
                float(config.other_parameters['distance']['third'])
            ]
        else:
            self._distance_to_Trigger = [74,76,88]  # m

        super().__init__("Hazard",
                         ego_vehicles,
                         config,
                         world,
                         randomize,
                         debug_mode,
                         criteria_enable=criteria_enable)

    def _initialize_actors(self, config):
        """
        Custom initialization
        """

        starting_wp = self._map.get_waypoint(config.trigger_points[0].location)
        if 'end_bycicle_distance' in config.other_parameters:
            self._end_bycicle_distance = float(
                config.other_parameters['end_bycicle_distance']['value'])
        else:
            self._end_bycicle_distance = 150
        self._target_location = starting_wp.next(self._end_bycicle_distance)[0].transform.location
        for offset,distance in zip(self._offset,self._distance_to_Trigger):

            bicycle_wps = starting_wp.next(distance)

            if not bicycle_wps:
                raise ValueError("Couldn't find a viable position to set up the bicycle actors")
            self._bicycle_wp.append(bicycle_wps[0])
            displacement = offset* bicycle_wps[0].lane_width / 2
            r_vec = bicycle_wps[0].transform.get_right_vector()
            w_loc = bicycle_wps[0].transform.location
            w_loc = w_loc + carla.Location(x=displacement * r_vec.x, y=displacement * r_vec.y)
            bycicle_transform = carla.Transform(w_loc, bicycle_wps[0].transform.rotation)
            bycicle = CarlaDataProvider.request_new_actor('vehicle.diamondback.century', bycicle_transform)
            self.other_actors.append(bycicle)

    def _create_behavior(self):
        """
        The vehicle has to drive the whole predetermined distance.
        """

        root = py_trees.composites.Sequence()
        if self.route_mode:
            total_dist = self._distance_to_Trigger[2] + 30
            root.add_child(LeaveSpaceInFront(total_dist))
            root.add_child(SwitchOutsideRouteLanesTest(False))
            root.add_child(ChangeOppositeBehavior(active=False))
        bycicle = py_trees.composites.Parallel(
                policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        bycicle.add_child(ConstantVelocityAgentBehavior(
                self.other_actors[2], self._target_location,target_speed=3.1,opt_dict={'offset':self._offset[2]* self._bicycle_wp[2].lane_width / 2}))

        bycicle.add_child(ConstantVelocityAgentBehavior(
                self.other_actors[1], self._target_location, target_speed=3,
                opt_dict={'offset': self._offset[1] * self._bicycle_wp[1].lane_width / 2}))
        bycicle.add_child(ConstantVelocityAgentBehavior(
                self.other_actors[0], self._target_location, target_speed=3,
                opt_dict={'offset': self._offset[0] * self._bicycle_wp[0].lane_width / 2}))
        root.add_child(bycicle)
        root.add_child(DriveDistance(self.ego_vehicles[0], self._drive_distance))
        if self.route_mode:
            root.add_child(SwitchOutsideRouteLanesTest(True))
            root.add_child(ChangeOppositeBehavior(active=True))
        root.add_child(ActorDestroy(self.other_actors[0]))
        root.add_child(ActorDestroy(self.other_actors[1]))
        root.add_child(ActorDestroy(self.other_actors[2]))

        return root

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        if self.route_mode:
            return []
        return [CollisionTest(self.ego_vehicles[0])]

    def __del__(self):
        """
        Remove all actors and traffic lights upon deletion
        """
        self.remove_all_actors()

class AccidentTwoWays(BasicScenario):
    """
    This class holds everything required for a scenario in which there is an accident
    in front of the ego, forcing it to react. A police vehicle is located before
    two other cars that have been in an accident.
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=180):
        """
        Setup all relevant parameters and create scenario
        and instantiate scenario manager
        """
        self._world = world
        self._map = CarlaDataProvider.get_map()
        self.timeout = timeout
        self._drive_distance = 120
        self._distance_to_accident = 100
        self._offset = 0.75
        self._first_distance = 10
        self._second_distance = 6
        self._accident_wp = None

        super().__init__("AccidentTwoWays",
                         ego_vehicles,
                         config,
                         world,
                         randomize,
                         debug_mode,
                         criteria_enable=criteria_enable)

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        starting_wp = self._map.get_waypoint(config.trigger_points[0].location)
        accident_wps = starting_wp.next(self._distance_to_accident)
        pre_accident_wps = starting_wp.next(self._distance_to_accident/2)
        if not accident_wps: 
            raise ValueError("Couldn't find a viable position to set up the accident actors")
        if not pre_accident_wps: 
            raise ValueError("Couldn't find a viable position to set up the accident actors")
        self._accident_wp = accident_wps[0]

        # Create the police vehicle
        displacement = self._offset * self._accident_wp.lane_width / 2
        r_vec = self._accident_wp.transform.get_right_vector()
        w_loc = self._accident_wp.transform.location
        w_loc += carla.Location(x=displacement * r_vec.x, y=displacement * r_vec.y)
        police_transform = carla.Transform(w_loc, self._accident_wp.transform.rotation)
        police_car = CarlaDataProvider.request_new_actor('vehicle.dodge.charger_police_2020', police_transform)
        police_lights = carla.VehicleLightState.Special1
        police_lights |= carla.VehicleLightState.Special2
        police_lights |= carla.VehicleLightState.Position
        police_car.set_light_state(carla.VehicleLightState(police_lights))
        self.other_actors.append(police_car)

        # Create the first vehicle that has been in the accident
        vehicle_wps = self._accident_wp.next(self._first_distance)
        if not vehicle_wps: 
            raise ValueError("Couldn't find a viable position to set up the accident actors")
        vehicle_wp = vehicle_wps[0]
        self._accident_wp = pre_accident_wps[0]
        displacement = self._offset * vehicle_wp.lane_width / 2
        r_vec = vehicle_wp.transform.get_right_vector()
        w_loc = vehicle_wp.transform.location
        w_loc += carla.Location(x=displacement * r_vec.x, y=displacement * r_vec.y)
        vehicle_1_transform = carla.Transform(w_loc, vehicle_wp.transform.rotation)
        vehicle_1_car = CarlaDataProvider.request_new_actor('vehicle.tesla.model3', vehicle_1_transform)
        self.other_actors.append(vehicle_1_car)

        # Create the second vehicle that has been in the accident
        vehicle_wps = vehicle_wp.next(self._second_distance)
        if not vehicle_wps: 
            raise ValueError("Couldn't find a viable position to set up the accident actors")
        vehicle_wp = vehicle_wps[0]

        displacement = self._offset * vehicle_wp.lane_width / 2
        r_vec = vehicle_wp.transform.get_right_vector()
        w_loc = vehicle_wp.transform.location
        w_loc += carla.Location(x=displacement * r_vec.x, y=displacement * r_vec.y)
        vehicle_2_transform = carla.Transform(w_loc, vehicle_wp.transform.rotation)
        vehicle_2_car = CarlaDataProvider.request_new_actor('vehicle.mercedes.coupe_2020', vehicle_2_transform)
        self.other_actors.append(vehicle_2_car)

    def _create_behavior(self):
        """
        The vehicle has to drive the whole predetermined distance.
        """
        root = py_trees.composites.Sequence()
        if self.route_mode:
            total_dist = self._distance_to_accident + self._first_distance + self._second_distance + 20
            root.add_child(LeaveSpaceInFront(total_dist))
            root.add_child(SwitchOutsideRouteLanesTest(False))
            root.add_child(ChangeOppositeBehavior(active=False))
        root.add_child(DriveDistance(self.ego_vehicles[0], self._drive_distance))
        if self.route_mode:
            root.add_child(SwitchOutsideRouteLanesTest(True))
            root.add_child(ChangeOppositeBehavior(active=True))
        root.add_child(ActorDestroy(self.other_actors[0]))
        root.add_child(ActorDestroy(self.other_actors[1]))
        root.add_child(ActorDestroy(self.other_actors[2]))

        return root

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        if self.route_mode:
            return []
        return [CollisionTest(self.ego_vehicles[0])]

    def __del__(self):
        """
        Remove all actors and traffic lights upon deletion
        """
        self.remove_all_actors()
