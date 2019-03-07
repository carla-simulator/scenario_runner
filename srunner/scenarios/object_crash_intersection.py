#!/usr/bin/env python

#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
"""
Object crash with prior vehicle action scenario:
The scenario realizes the user controlled ego vehicle
moving along the road and encounters a cyclist ahead after taking a right or left turn.
"""

import py_trees
import carla

from srunner.scenariomanager.atomic_scenario_behavior import *
from srunner.scenariomanager.atomic_scenario_criteria import *
from srunner.scenariomanager.timer import TimeOut
from srunner.scenarios.basic_scenario import *

VEHICLE_TURNING_SCENARIOS = [
    "VehicleTurningRight",
    "VehicleTurningLeft"
]


class VehicleTurningRight(BasicScenario):
    """
    This class holds everything required for a simple object crash
    with prior vehicle action involving a vehicle and a cyclist.
    The ego vehicle is passing through a road and encounters
    a cyclist after taking a right turn.
    """
    category = "VehicleTurning"
    timeout = 90

    # other vehicle parameters
    _other_actor_target_velocity = 10

    def __init__(self, world, ego_vehicle, config, randomize=False, debug_mode=False):
        """
        Setup all relevant parameters and create scenario
        """

        super(VehicleTurningRight, self).__init__("VehicleTurningRight",
                                                  ego_vehicle,
                                                  config,
                                                  world,
                                                  debug_mode)

    @staticmethod
    def initialize_actors(ego_vehicle):
        """
        This method returns the list of participant actors and their initial positions for the scenario
        """
        actor_parameters = []
        world = ego_vehicle.get_world()
        wmap = world.get_map()
        waypoint = wmap.get_waypoint(ego_vehicle.get_location())
        _wp = generate_target_waypoint(waypoint, -1)
        model = 'vehicle.diamondback.century'
        offset = {"orientation": 270, "position": 90, "z": 0.2, "k": 0.7}
        _wp = _wp.next(10)[-1]
        lane_width = _wp.lane_width
        location = _wp.transform.location
        orientation_yaw = _wp.transform.rotation.yaw+offset["orientation"]
        position_yaw = _wp.transform.rotation.yaw+offset["position"]
        offset_location = carla.Location(
            offset['k']*lane_width*math.cos(math.radians(position_yaw)),
            offset['k']*lane_width*math.sin(math.radians(position_yaw)))
        location += offset_location
        location.z += offset["z"]
        transform = carla.Transform(location, carla.Rotation(yaw=orientation_yaw))
        actor_parameters.append((model, transform))

        return actor_parameters

    def _create_behavior(self):
        """
        After invoking this scenario, cyclist will wait for the user
        controlled vehicle to enter the in the trigger distance region,
        the cyclist starts crossing the road once the condition meets,
        ego vehicle has to avoid the crash after a right turn, but
        continue driving after the road is clear.If this does not happen
        within 90 seconds, a timeout stops the scenario.
        """
        lane_width = self.ego_vehicle.get_world().get_map().get_waypoint(self.ego_vehicle.get_location()).lane_width
        lane_width = lane_width+(1.10*lane_width)

        trigger_distance = InTriggerDistanceToVehicle(self.other_actors[0], self.ego_vehicle, 20)
        actor_velocity = KeepVelocity(self.other_actors[0], self._other_actor_target_velocity)
        actor_traverse = DriveDistance(self.other_actors[0], 0.30*lane_width)
        actor_brakes = StopVehicle(self.other_actors[0], 1)
        actor_stand = TimeOut(10)
        post_timer_velocity_actor = KeepVelocity(self.other_actors[0], self._other_actor_target_velocity)
        post_timer_traverse_actor = DriveDistance(self.other_actors[0], 0.70*lane_width)
        end_condition = TimeOut(5)

        # non leaf nodes
        root = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        scenario_sequence = py_trees.composites.Sequence()
        actor_ego_sync = py_trees.composites.Parallel(
            "Synchronization of actor and ego vehicle",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        after_timer_actor = py_trees.composites.Parallel(
            "After timout actor will cross the remaining lane_width",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        # building the tree
        root.add_child(scenario_sequence)
        scenario_sequence.add_child(trigger_distance)
        scenario_sequence.add_child(actor_ego_sync)
        scenario_sequence.add_child(actor_brakes)
        scenario_sequence.add_child(actor_stand)
        scenario_sequence.add_child(after_timer_actor)
        scenario_sequence.add_child(end_condition)

        actor_ego_sync.add_child(actor_velocity)
        actor_ego_sync.add_child(actor_traverse)

        after_timer_actor.add_child(post_timer_velocity_actor)
        after_timer_actor.add_child(post_timer_traverse_actor)
        return root

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []
        collision_criterion = CollisionTest(self.ego_vehicle)
        criteria.append(collision_criterion)
        return criteria

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()


class VehicleTurningLeft(BasicScenario):

    """
    This class holds everything required for a simple object crash
    with prior vehicle action involving a vehicle and a cyclist.
    The ego vehicle is passing through a road and encounters
    a cyclist after taking a left turn.
    """

    category = "VehicleTurning"

    timeout = 90

    # other vehicle parameters
    _other_actor_target_velocity = 10

    _location_of_collision = carla.Location(x=88.6, y=75.8, z=38)

    def __init__(self, world, ego_vehicle, config, randomize=False, debug_mode=False):
        """
        Setup all relevant parameters and create scenario
        """

        super(VehicleTurningLeft, self).__init__("VehicleTurningLeft",
                                                 ego_vehicle,
                                                 config,
                                                 world,
                                                 debug_mode)

    @staticmethod
    def initialize_actors(ego_vehicle):
        """
        This method returns the list of participant actors and their initial positions for the scenario
        """
        actor_parameters = []
        wmap = ego_vehicle.get_world().get_map()
        lane_width = wmap.get_waypoint(ego_vehicle.get_location()).lane_width
        waypoint = wmap.get_waypoint(ego_vehicle.get_location())
        _wp = generate_target_waypoint(waypoint, turn=1)
        model = 'vehicle.diamondback.century'
        offset = {"orientation": 270, "position": 80, "z": 0.2, "k": 0.7}
        _wp = _wp.next(10)[-1]
        lane_width = _wp.lane_width
        location = _wp.transform.location
        orientation_yaw = _wp.transform.rotation.yaw+offset["orientation"]
        position_yaw = _wp.transform.rotation.yaw+offset["position"]
        offset_location = carla.Location(
            offset['k']*lane_width*math.cos(math.radians(position_yaw)),
            offset['k']*lane_width*math.sin(math.radians(position_yaw)))
        location += offset_location
        location.z += offset["z"]
        transform = carla.Transform(location, carla.Rotation(yaw=orientation_yaw))
        actor_parameters.append((model, transform))

        return actor_parameters

    def _create_behavior(self):
        """
        After invoking this scenario, cyclist will wait for the user
        controlled vehicle to enter the in the trigger distance region,
        the cyclist starts crossing the road once the condition meets,
        ego vehicle has to avoid the crash after a left turn, but
        continue driving after the road is clear.If this does not happen
        within 90 seconds, a timeout stops the scenario.
        """
        lane_width = self.ego_vehicle.get_world().get_map().get_waypoint(self.ego_vehicle.get_location()).lane_width
        lane_width = lane_width+(1.10*lane_width)
        trigger_distance = InTriggerDistanceToVehicle(self.other_actors[0], self.ego_vehicle, 25)
        actor_velocity = KeepVelocity(self.other_actors[0], self._other_actor_target_velocity)
        actor_traverse = DriveDistance(self.other_actors[0], 0.30*lane_width)
        actor_brakes = StopVehicle(self.other_actors[0], 1)
        actor_stand = TimeOut(10)
        post_timer_velocity_actor = KeepVelocity(self.other_actors[0], self._other_actor_target_velocity)
        post_timer_traverse_actor = DriveDistance(self.other_actors[0], 0.70*lane_width)
        end_condition = TimeOut(5)

        # non leaf nodes
        root = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        scenario_sequence = py_trees.composites.Sequence()

        actor_ego_sync = py_trees.composites.Parallel(
            "Synchronization of actor and ego vehicle",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        after_timer_actor = py_trees.composites.Parallel(
            "After timout actor will cross the remaining lane_width",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        # building the tree
        root.add_child(scenario_sequence)
        scenario_sequence.add_child(trigger_distance)
        scenario_sequence.add_child(actor_ego_sync)
        scenario_sequence.add_child(actor_brakes)
        scenario_sequence.add_child(actor_stand)
        scenario_sequence.add_child(after_timer_actor)
        scenario_sequence.add_child(end_condition)

        actor_ego_sync.add_child(actor_velocity)
        actor_ego_sync.add_child(actor_traverse)

        after_timer_actor.add_child(post_timer_velocity_actor)
        after_timer_actor.add_child(post_timer_traverse_actor)
        return root

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []
        collision_criterion = CollisionTest(self.ego_vehicle)

        criteria.append(collision_criterion)
        return criteria

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()
