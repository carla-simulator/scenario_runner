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
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import DriveDistance, InTriggerDistanceToLocation
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.background_manager import LeaveSpaceInFront, ChangeOppositeBehavior, SetMaxSpeed


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

        self._lights = carla.VehicleLightState.Special1 | carla.VehicleLightState.Special2

        if 'speed' in config.other_parameters:
            self._max_speed = float(config.other_parameters['speed']['value'])
        else:
            self._max_speed = 60

        if 'timeout' in config.other_parameters:
            self._scenario_timeout = float(config.other_parameters['flow_distance']['value'])
        else:
            self._scenario_timeout = 180

        super().__init__(
            "Accident", ego_vehicles, config, world, randomize, debug_mode, criteria_enable=criteria_enable)

    def _move_waypoint_forward(self, wp, distance):
        next_wps = wp.next(distance)
        if not next_wps:
            raise ValueError("Couldn't find a viable position to set up an accident actor")
        return next_wps[0]


    def _spawn_obstacle(self, wp, blueprint, attributes=None):

        displacement = self._offset * wp.lane_width / 2
        r_vec = wp.transform.get_right_vector()
        if self._direction == 'left':
            r_vec *= -1

        spawn_transform = wp.transform
        spawn_transform.location += carla.Location(x=displacement * r_vec.x, y=displacement * r_vec.y)
        actor = CarlaDataProvider.request_new_actor(blueprint, spawn_transform, attribute_filter=attributes)
        if not actor:
            raise ValueError("Couldn't spawn an obstacle actor")

        return actor

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        starting_wp = self._map.get_waypoint(config.trigger_points[0].location)

        # Spawn the police vehicle
        self._accident_wp = self._move_waypoint_forward(starting_wp, self._distance)
        police_car = self._spawn_obstacle(self._accident_wp, 'vehicle.dodge.charger_police_2020')

        # Set its initial conditions
        lights = police_car.get_light_state()
        lights |= self._lights
        police_car.set_light_state(carla.VehicleLightState(lights))
        police_car.apply_control(carla.VehicleControl(hand_brake=True))
        self.other_actors.append(police_car)

        # Create the first vehicle that has been in the accident
        first_vehicle_wp = self._move_waypoint_forward(self._accident_wp, self._first_distance)
        first_actor = self._spawn_obstacle(first_vehicle_wp, 'vehicle.*', {'base_type': 'car', 'has_lights': False})

        # Set its initial conditions
        first_actor.apply_control(carla.VehicleControl(hand_brake=True))
        self.other_actors.append(first_actor)

        # Create the second vehicle that has been in the accident
        second_vehicle_wp = self._move_waypoint_forward(first_vehicle_wp, self._second_distance)
        second_actor = self._spawn_obstacle(second_vehicle_wp, 'vehicle.*', {'base_type': 'car', 'has_lights': False})

        # Set its initial conditions
        second_actor.apply_control(carla.VehicleControl(hand_brake=True))
        self.other_actors.append(second_actor)

    def _create_behavior(self):
        """
        The vehicle has to drive the whole predetermined distance.
        """
        total_dist = self._distance + self._first_distance + self._second_distance + 20

        root = py_trees.composites.Sequence()
        if self.route_mode:
            root.add_child(LeaveSpaceInFront(total_dist))
            root.add_child(SetMaxSpeed(self._max_speed))

        end_condition = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        end_condition.add_child(DriveDistance(self.ego_vehicles[0], self._drive_distance))
        end_condition.add_child(ScenarioTimeout(self._scenario_timeout, self.config.name))
        root.add_child(end_condition)
        root.add_child(ActorDestroy(self.other_actors[0]))
        root.add_child(ActorDestroy(self.other_actors[1]))
        root.add_child(ActorDestroy(self.other_actors[2]))

        if self.route_mode:
            root.add_child(SetMaxSpeed(0))
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
        self._trigger_distance = 30
        self._drive_distance = self._trigger_distance + self._takeover_max_dist

        root = py_trees.composites.Sequence()
        if self.route_mode:
            total_dist = self._distance + self._first_distance + self._second_distance + 20
            root.add_child(LeaveSpaceInFront(total_dist))

        timeout_parallel = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        timeout_parallel.add_child(ScenarioTimeout(self._scenario_timeout, self.config.name))

        behavior = py_trees.composites.Sequence()
        behavior.add_child(InTriggerDistanceToLocation(
            self.ego_vehicles[0], self._accident_wp.transform.location, self._trigger_distance))

        if self.route_mode:
            behavior.add_child(SwitchWrongDirectionTest(False))
            behavior.add_child(ChangeOppositeBehavior(spawn_dist=self._opposite_frequency))
        behavior.add_child(DriveDistance(self.ego_vehicles[0], self._drive_distance))

        timeout_parallel.add_child(behavior)

        root.add_child(timeout_parallel)

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

        if 'speed' in config.other_parameters:
            self._max_speed = float(config.other_parameters['speed']['value'])
        else:
            self._max_speed = 60

        if 'direction' in config.other_parameters:
            self._direction = config.other_parameters['direction']['value']
        else:
            self._direction = 'right'
        if self._direction not in ('left', 'right'):
            raise ValueError(f"'direction' must be either 'right' or 'left' but {self._direction} was given")

        super().__init__(
            "ParkedObstacle", ego_vehicles, config, world, randomize, debug_mode, criteria_enable=criteria_enable)

    def _move_waypoint_forward(self, wp, distance):
        next_wps = wp.next(distance)
        if not next_wps:
            raise ValueError("Couldn't find a viable position to set up an accident actor")
        return next_wps[0]


    def _spawn_obstacle(self, wp, blueprint, attributes=None):
        displacement = self._offset * wp.lane_width / 2
        r_vec = wp.transform.get_right_vector()
        if self._direction == 'left':
            r_vec *= -1
        spawn_transform = wp.transform
        spawn_transform.location += carla.Location(x=displacement * r_vec.x, y=displacement * r_vec.y)
        actor = CarlaDataProvider.request_new_actor(blueprint, spawn_transform, attribute_filter=attributes)
        if not actor:
            raise ValueError("Couldn't spawn an obstacle actor")

        return actor

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        starting_wp = self._map.get_waypoint(config.trigger_points[0].location)

        # Create the first vehicle that has been in the accident
        self._vehicle_wp = self._move_waypoint_forward(starting_wp, self._distance)
        parked_actor = self._spawn_obstacle(self._vehicle_wp, 'vehicle.*', {'base_type': 'car', 'has_lights': True})

        lights = parked_actor.get_light_state()
        lights |= self._lights
        parked_actor.set_light_state(carla.VehicleLightState(lights))
        parked_actor.apply_control(carla.VehicleControl(hand_brake=True))
        self.other_actors.append(parked_actor)


    def _create_behavior(self):
        """
        The vehicle has to drive the whole predetermined distance.
        """
        total_dist = self._distance + 20

        root = py_trees.composites.Sequence()
        if self.route_mode:
            root.add_child(LeaveSpaceInFront(total_dist))
            root.add_child(SetMaxSpeed(self._max_speed))
        end_condition = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        end_condition.add_child(DriveDistance(self.ego_vehicles[0], self._drive_distance))
        end_condition.add_child(ScenarioTimeout(self._scenario_timeout, self.config.name))
        root.add_child(end_condition)
        root.add_child(ActorDestroy(self.other_actors[0]))
        if self.route_mode:
            root.add_child(SetMaxSpeed(0))

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
        self._trigger_distance = 30
        self._drive_distance = self._trigger_distance + 40

        root = py_trees.composites.Sequence()
        if self.route_mode:
            total_dist = self._distance + 20
            root.add_child(LeaveSpaceInFront(total_dist))

        timeout_parallel = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        timeout_parallel.add_child(ScenarioTimeout(self._scenario_timeout, self.config.name))

        behavior = py_trees.composites.Sequence()
        behavior.add_child(InTriggerDistanceToLocation(
            self.ego_vehicles[0], self._vehicle_wp.transform.location, self._trigger_distance))

        if self.route_mode:
            behavior.add_child(SwitchWrongDirectionTest(False))
            behavior.add_child(ChangeOppositeBehavior(spawn_dist=self._opposite_frequency))
        behavior.add_child(DriveDistance(self.ego_vehicles[0], self._drive_distance))

        timeout_parallel.add_child(behavior)

        root.add_child(timeout_parallel)

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

        for offset, distance in zip(self._offset, self._distance_to_Trigger):

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
