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
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorDestroy,
                                                                      SwitchWrongDirectionTest,
                                                                      ConstantVelocityAgentBehavior,
                                                                      ScenarioTimeout)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest, ScenarioTimeoutTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import DriveDistance
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.background_manager import LeaveSpaceInFront, ChangeOppositeBehavior


class Accident(BasicScenario):
    """
    This class holds everything required for a scenario in which there is an accident
    in front of the ego, forcing it to lane change. A police vehicle is located before
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
        if 'distance' in config.other_parameters:
            self._distance = int(config.other_parameters['distance']['value'])
        else:
            self._distance = 120

        if 'direction' in config.other_parameters:
            self._direction = config.other_parameters['direction']['value']
        else:
            self._direction = 'right'
        if self._direction not in ('left', 'right'):
            raise ValueError(f"'direction' must be either 'right' or 'left' but {self._direction} was given")

        self._offset = 0.75
        self._first_distance = 10
        self._second_distance = 6

        self._takeover_max_dist = self._first_distance + self._second_distance + 40
        self._drive_distance = self._distance + self._takeover_max_dist

        self._accident_wp = None

        self._lights = carla.VehicleLightState.Special1 | carla.VehicleLightState.Special2

        if 'timeout' in config.other_parameters:
            self._scenario_timeout = float(config.other_parameters['flow_distance']['value'])
        else:
            self._scenario_timeout = 180

        super().__init__(
            "Accident", ego_vehicles, config, world, randomize, debug_mode, criteria_enable=criteria_enable)

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        starting_wp = self._map.get_waypoint(config.trigger_points[0].location)
        accident_wps = starting_wp.next(self._distance)
        pre_accident_wps = starting_wp.next(self._distance / 2)
        if not accident_wps: 
            raise ValueError("Couldn't find a viable position to set up the accident actors")
        if not pre_accident_wps: 
            raise ValueError("Couldn't find a viable position to set up the accident actors")
        self._accident_wp = accident_wps[0]

        # Create the police vehicle
        displacement = self._offset * self._accident_wp.lane_width / 2
        r_vec = self._accident_wp.transform.get_right_vector()
        if self._direction == 'left':
            r_vec *= -1
        w_loc = self._accident_wp.transform.location
        w_loc += carla.Location(x=displacement * r_vec.x, y=displacement * r_vec.y)
        police_transform = carla.Transform(w_loc, self._accident_wp.transform.rotation)
        police_car = CarlaDataProvider.request_new_actor('vehicle.dodge.charger_police_2020', police_transform)
        if not police_car:
            raise ValueError("Couldn't spawn the police car")
        lights = police_car.get_light_state()
        lights |= self._lights
        police_car.set_light_state(carla.VehicleLightState(lights))
        police_car.apply_control(carla.VehicleControl(hand_brake=True))
        self.other_actors.append(police_car)

        # Create the first vehicle that has been in the accident
        vehicle_wps = self._accident_wp.next(self._first_distance)
        if not vehicle_wps: 
            raise ValueError("Couldn't find a viable position to set up the accident actors")
        vehicle_wp = vehicle_wps[0]
        self._accident_wp = pre_accident_wps[0]
        displacement = self._offset * vehicle_wp.lane_width / 2
        r_vec = vehicle_wp.transform.get_right_vector()
        if self._direction == 'left':
            r_vec *= -1
        w_loc = vehicle_wp.transform.location
        w_loc += carla.Location(x=displacement * r_vec.x, y=displacement * r_vec.y)
        vehicle_1_transform = carla.Transform(w_loc, vehicle_wp.transform.rotation)
        vehicle_1_car = CarlaDataProvider.request_new_actor(
            'vehicle.*', vehicle_1_transform, attribute_filter={'base_type': 'car', 'has_lights': False})
        if not vehicle_1_car:
            raise ValueError("Couldn't spawn the accident car")
        vehicle_1_car.apply_control(carla.VehicleControl(hand_brake=True))
        self.other_actors.append(vehicle_1_car)

        # Create the second vehicle that has been in the accident
        vehicle_wps = vehicle_wp.next(self._second_distance)
        if not vehicle_wps:
            raise ValueError("Couldn't find a viable position to set up the accident actors")
        vehicle_wp = vehicle_wps[0]

        displacement = self._offset * vehicle_wp.lane_width / 2
        r_vec = vehicle_wp.transform.get_right_vector()
        if self._direction == 'left':
            r_vec *= -1
        w_loc = vehicle_wp.transform.location
        w_loc += carla.Location(x=displacement * r_vec.x, y=displacement * r_vec.y)
        vehicle_2_transform = carla.Transform(w_loc, vehicle_wp.transform.rotation)
        vehicle_2_car = CarlaDataProvider.request_new_actor(
            'vehicle.*', vehicle_2_transform, attribute_filter={'base_type': 'car', 'has_lights': False})
        if not vehicle_2_car:
            raise ValueError("Couldn't spawn the accident car")
        vehicle_2_car.apply_control(carla.VehicleControl(hand_brake=True))
        self.other_actors.append(vehicle_2_car)

    def _create_behavior(self):
        """
        The vehicle has to drive the whole predetermined distance.
        """
        total_dist = self._distance + self._first_distance + self._second_distance + 20

        root = py_trees.composites.Sequence()
        if self.route_mode:
            root.add_child(LeaveSpaceInFront(total_dist))

        end_condition = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        end_condition.add_child(DriveDistance(self.ego_vehicles[0], self._drive_distance))
        end_condition.add_child(ScenarioTimeout(self._scenario_timeout, self.config.name))
        root.add_child(end_condition)
        root.add_child(ActorDestroy(self.other_actors[0]))
        root.add_child(ActorDestroy(self.other_actors[1]))
        root.add_child(ActorDestroy(self.other_actors[2]))
        return root

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = [ScenarioTimeoutTest(self.ego_vehicles[0], self.config.name)]
        if not self.route_mode:
            criteria.append(CollisionTest(self.ego_vehicles[0]))
        return criteria

    def __del__(self):
        """
        Remove all actors and traffic lights upon deletion
        """
        self.remove_all_actors()


class AccidentTwoWays(Accident):
    """
    Variation of the Accident scenario but the ego now has to invade the opposite lane
    """
    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True, timeout=180):
        if 'frequency' in config.other_parameters:
            self._opposite_frequency = float(config.other_parameters['frequency']['value'])
        else:
            self._opposite_frequency = 200
        super().__init__(world, ego_vehicles, config, randomize, debug_mode, criteria_enable, timeout)


    def _create_behavior(self):
        """
        The vehicle has to drive the whole predetermined distance. Adapt the opposite flow to
        let the ego invade the opposite lane.
        """
        total_dist = self._distance + self._first_distance + self._second_distance + 20

        root = py_trees.composites.Sequence()
        if self.route_mode:
            root.add_child(LeaveSpaceInFront(total_dist))
            root.add_child(SwitchWrongDirectionTest(False))
            root.add_child(ChangeOppositeBehavior(spawn_dist=self._opposite_frequency))
        end_condition = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        end_condition.add_child(DriveDistance(self.ego_vehicles[0], self._drive_distance))
        end_condition.add_child(ScenarioTimeout(self._scenario_timeout, self.config.name))
        root.add_child(end_condition)
        if self.route_mode:
            root.add_child(SwitchWrongDirectionTest(True))
            root.add_child(ChangeOppositeBehavior(spawn_dist=50))
        root.add_child(ActorDestroy(self.other_actors[0]))
        root.add_child(ActorDestroy(self.other_actors[1]))
        root.add_child(ActorDestroy(self.other_actors[2]))

        return root


class ParkedObstacle(BasicScenario):
    """
    Scenarios in which a parked vehicle is incorrectly parked,
    forcing the ego to lane change out of the route's lane
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
        if 'distance' in config.other_parameters:
            self._distance = int(config.other_parameters['distance']['value'])
        else:
            self._distance = 120
        self._drive_distance = self._distance + 20
        self._offset = 1.0

        self._lights = carla.VehicleLightState.RightBlinker | carla.VehicleLightState.LeftBlinker

        if 'timeout' in config.other_parameters:
            self._scenario_timeout = float(config.other_parameters['flow_distance']['value'])
        else:
            self._scenario_timeout = 180

        super().__init__(
            "ParkedObstacle", ego_vehicles, config, world, randomize, debug_mode, criteria_enable=criteria_enable)

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        starting_wp = self._map.get_waypoint(config.trigger_points[0].location)
        parked_wps = starting_wp.next(self._distance)
        if not parked_wps: 
            raise ValueError("Couldn't find a viable position to set up the accident actors")
        self._parked_wp = parked_wps[0]

        # Create the parked vehicle
        displacement = self._offset * self._parked_wp.lane_width / 2
        r_vec = self._parked_wp.transform.get_right_vector()
        w_loc = self._parked_wp.transform.location
        w_loc += carla.Location(x=displacement * r_vec.x, y=displacement * r_vec.y)
        parked_transform = carla.Transform(w_loc, self._parked_wp.transform.rotation)
        parked_car = CarlaDataProvider.request_new_actor(
            'vehicle.*', parked_transform, attribute_filter={'base_type': 'car', 'has_lights': True})
        if not parked_car:
            raise ValueError("Couldn't spawn the parked car")
        self.other_actors.append(parked_car)

        lights = parked_car.get_light_state()
        lights |= self._lights
        parked_car.set_light_state(carla.VehicleLightState(lights))
        parked_car.apply_control(carla.VehicleControl(hand_brake=True))

        pre_parked_wps = starting_wp.next(self._distance / 2)
        if not pre_parked_wps: 
            raise ValueError("Couldn't find a viable position to set up the accident actors")
        self._pre_parked_wp = pre_parked_wps[0]


    def _create_behavior(self):
        """
        The vehicle has to drive the whole predetermined distance.
        """
        total_dist = self._distance + 20

        root = py_trees.composites.Sequence()
        if self.route_mode:
            root.add_child(LeaveSpaceInFront(total_dist))
        end_condition = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        end_condition.add_child(DriveDistance(self.ego_vehicles[0], self._drive_distance))
        end_condition.add_child(ScenarioTimeout(self._scenario_timeout, self.config.name))
        root.add_child(end_condition)
        root.add_child(ActorDestroy(self.other_actors[0]))

        return root

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = [ScenarioTimeoutTest(self.ego_vehicles[0], self.config.name)]
        if not self.route_mode:
            criteria.append(CollisionTest(self.ego_vehicles[0]))
        return criteria

    def __del__(self):
        """
        Remove all actors and traffic lights upon deletion
        """
        self.remove_all_actors()


class ParkedObstacleTwoWays(ParkedObstacle):
    """
    Variation of the ParkedObstacle scenario but the ego now has to invade the opposite lane
    """
    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True, timeout=180):
        if 'frequency' in config.other_parameters:
            self._opposite_frequency = float(config.other_parameters['frequency']['value'])
        else:
            self._opposite_frequency = 200
        super().__init__(world, ego_vehicles, config, randomize, debug_mode, criteria_enable, timeout)

    def _create_behavior(self):
        """
        The vehicle has to drive the whole predetermined distance. Adapt the opposite flow to
        let the ego invade the opposite lane.
        """
        total_dist = self._distance + 20

        root = py_trees.composites.Sequence()
        if self.route_mode:
            root.add_child(LeaveSpaceInFront(total_dist))
            root.add_child(SwitchWrongDirectionTest(False))
            root.add_child(ChangeOppositeBehavior(spawn_dist=self._opposite_frequency))
        end_condition = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        end_condition.add_child(DriveDistance(self.ego_vehicles[0], self._drive_distance))
        end_condition.add_child(ScenarioTimeout(self._scenario_timeout, self.config.name))
        root.add_child(end_condition)
        if self.route_mode:
            root.add_child(SwitchWrongDirectionTest(True))
            root.add_child(ChangeOppositeBehavior(spawn_dist=50))
        root.add_child(ActorDestroy(self.other_actors[0]))

        return root


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
        self._offset = [0.6, 0.75, 0.9]
        self._bicycle_wp = []
        self._target_location = None
        self._plan = []
        self._bicycle_speed = 3  # m/s

        if 'distance' in config.other_parameters:
            self._distance_to_Trigger = [
                float(config.other_parameters['distance']['first']),
                float(config.other_parameters['distance']['second']),
                float(config.other_parameters['distance']['third'])
            ]
        else:
            self._distance_to_Trigger = [74,76,88]  # m

        if 'timeout' in config.other_parameters:
            self._scenario_timeout = float(config.other_parameters['flow_distance']['value'])
        else:
            self._scenario_timeout = 180

        super().__init__("BicycleFlowAtSideLane",
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
            root.add_child(SwitchWrongDirectionTest(False))
            root.add_child(ChangeOppositeBehavior(active=False))
        bycicle = py_trees.composites.Parallel(
                policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        bycicle.add_child(ConstantVelocityAgentBehavior(
                self.other_actors[2], self._target_location, target_speed = self._bicycle_speed,
                opt_dict={'offset': self._offset[2] * self._bicycle_wp[2].lane_width / 2}))
        bycicle.add_child(ConstantVelocityAgentBehavior(
                self.other_actors[1], self._target_location, target_speed = self._bicycle_speed,
                opt_dict={'offset': self._offset[1] * self._bicycle_wp[1].lane_width / 2}))
        bycicle.add_child(ConstantVelocityAgentBehavior(
                self.other_actors[0], self._target_location, target_speed = self._bicycle_speed,
                opt_dict={'offset': self._offset[0] * self._bicycle_wp[0].lane_width / 2}))
        root.add_child(bycicle)
        end_condition = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        end_condition.add_child(DriveDistance(self.ego_vehicles[0], self._drive_distance))
        end_condition.add_child(ScenarioTimeout(self._scenario_timeout, self.config.name))
        root.add_child(end_condition)
        if self.route_mode:
            root.add_child(SwitchWrongDirectionTest(True))
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
        criteria = [ScenarioTimeoutTest(self.ego_vehicles[0], self.config.name)]
        if not self.route_mode:
            criteria.append(CollisionTest(self.ego_vehicles[0]))
        return criteria

    def __del__(self):
        """
        Remove all actors and traffic lights upon deletion
        """
        self.remove_all_actors()
