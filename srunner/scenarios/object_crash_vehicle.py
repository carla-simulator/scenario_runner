#!/usr/bin/env python
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Object crash without prior vehicle action scenario:
The scenario realizes the user controlled ego vehicle
moving along the road and encountering a cyclist ahead.
"""

import py_trees

from srunner.scenariomanager.atomic_scenario_behavior import *
from srunner.scenariomanager.atomic_scenario_criteria import *
from srunner.scenariomanager.timer import TimeOut
from srunner.scenarios.basic_scenario import *
from srunner.scenarios.scenario_helper import *

OBJECT_CROSSING_SCENARIOS = [
    "StationaryObjectCrossing",
    "DynamicObjectCrossing"
]


class StationaryObjectCrossing(BasicScenario):

    """
    This class holds everything required for a simple object crash
    without prior vehicle action involving a vehicle and a cyclist.
    The ego vehicle is passing through a road and encounters
    a stationary cyclist.
    """

    category = "ObjectCrossing"

    timeout = 60

    # ego vehicle parameters
    _ego_vehicle_velocity_allowed = 20
    _ego_vehicle_distance_to_other = 35

    def __init__(self, world, ego_vehicle, config, randomize=False, debug_mode=False, criteria_enable=True):
        """
        Setup all relevant parameters and create scenario
        """
        self._wmap = world.get_map()
        self._reference_waypoint = self._wmap.get_waypoint(config.ego_vehicle.transform.location)

        # other vehicle parameters
        self._other_actor_target_velocity = 10

        super(StationaryObjectCrossing, self).__init__("Stationaryobjectcrossing",
                                                       ego_vehicle,
                                                       config,
                                                       world,
                                                       debug_mode,
                                                       criteria_enable=criteria_enable)

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        _start_distance = 40
        lane_width = self._reference_waypoint.lane_width
        location, _ = get_location_in_distance(self.ego_vehicle, _start_distance)
        waypoint = self._wmap.get_waypoint(location)
        offset = {"orientation": 270, "position": 90, "z": 0.2, "k": 0.2}
        position_yaw = waypoint.transform.rotation.yaw + offset['position']
        orientation_yaw = waypoint.transform.rotation.yaw + offset['orientation']
        offset_location = carla.Location(
            offset['k']*lane_width*math.cos(math.radians(position_yaw)),
            offset['k']*lane_width*math.sin(math.radians(position_yaw)))
        location += offset_location
        location.z += offset['z']
        transform = carla.Transform(location, carla.Rotation(yaw=orientation_yaw))
        static = CarlaActorPool.request_new_actor('static.prop.container', transform)
        static.set_simulate_physics(True)
        self.other_actors.append(static)

    def _create_behavior(self):
        """
        Only behavior here is to wait
        """
        lane_width = self.ego_vehicle.get_world().get_map().get_waypoint(self.ego_vehicle.get_location()).lane_width
        lane_width = lane_width+(1.25*lane_width)

        # leaf nodes
        start_condition = InTriggerDistanceToVehicle(self.other_actors[0], self.ego_vehicle, 10)
        actor_stand = TimeOut(30)
        actor_removed = ActorDestroy(self.other_actors[0])
        end_condition = DriveDistance(self.ego_vehicle, 50)

        # non leaf nodes
        root = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        scenario_sequence = py_trees.composites.Sequence()

        # building tree
        root.add_child(scenario_sequence)
        scenario_sequence.add_child(start_condition)
        scenario_sequence.add_child(actor_stand)
        scenario_sequence.add_child(actor_removed)
        scenario_sequence.add_child(end_condition)

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


class DynamicObjectCrossing(BasicScenario):
    """
    This class holds everything required for a simple object crash
    without prior vehicle action involving a vehicle and a cyclist,
    The ego vehicle is passing through a road,
    And encounters a cyclist crossing the road.
    """

    category = "ObjectCrossing"

    timeout = 60

    # ego vehicle parameters
    _ego_vehicle_velocity_allowed = 10
    _ego_vehicle_distance_driven = 50

    # other vehicle parameters
    _other_actor_target_velocity = 10
    _other_actor_max_brake = 1.0
    _time_to_reach = 12

    def __init__(self, world, ego_vehicle, config, randomize=False, debug_mode=False, criteria_enable=True):
        """
        Setup all relevant parameters and create scenario
        """
        self._wmap = world.get_map()
        self.category = "ObjectCrossing"
        self.timeout = 60
        self._reference_waypoint = self._wmap.get_waypoint(config.ego_vehicle.transform.location)

        # other vehicle parameters
        self._other_actor_target_velocity = 10
        self._other_actor_max_brake = 1.0
        self._time_to_reach = 12

        super(DynamicObjectCrossing, self).__init__("Dynamicobjectcrossing",
                                                    ego_vehicle,
                                                    config,
                                                    world,
                                                    debug_mode,
                                                    criteria_enable=criteria_enable)

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        # cyclist transform
        _start_distance = 40
        lane_width = self._reference_waypoint.lane_width
        location, _ = get_location_in_distance(self.ego_vehicle, _start_distance)
        waypoint = self._wmap.get_waypoint(location)
        offset = {"orientation": 270, "position": 90, "z": 0.2, "k": 1.1}
        position_yaw = waypoint.transform.rotation.yaw + offset['position']
        orientation_yaw = waypoint.transform.rotation.yaw + offset['orientation']
        offset_location = carla.Location(
            offset['k']*lane_width*math.cos(math.radians(position_yaw)),
            offset['k']*lane_width*math.sin(math.radians(position_yaw)))
        location += offset_location
        location.z += offset['z']
        transform = carla.Transform(location, carla.Rotation(yaw=orientation_yaw))
        first_vehicle = CarlaActorPool.request_new_actor('vehicle.diamondback.century', transform)
        self.other_actors.append(first_vehicle)

        # static object transform
        shift = 0.8
        x_ego = self.ego_vehicle.get_location().x
        y_ego = self.ego_vehicle.get_location().y
        x_cycle = transform.location.x
        y_cycle = transform.location.y
        x_static = x_ego + shift * (x_cycle - x_ego)
        y_static = y_ego + shift * (y_cycle - y_ego)

        transform2 = carla.Transform(carla.Location(x_static, y_static, transform.location.z+0.01))
        static = CarlaActorPool.request_new_actor('static.prop.vendingmachine', transform2)
        static.set_simulate_physics(True)
        self.other_actors.append(static)

    def _create_behavior(self):
        """
        After invoking this scenario, cyclist will wait for the user
        controlled vehicle to enter trigger distance region,
        the cyclist starts crossing the road once the condition meets,
        then after 60 seconds, a timeout stops the scenario
        """

        lane_width = self.ego_vehicle.get_world().get_map().get_waypoint(self.ego_vehicle.get_location()).lane_width
        lane_width = lane_width+(1.25*lane_width)

        # leaf nodes
        start_condition = InTimeToArrivalToVehicle(self.other_actors[0], self.ego_vehicle, self._time_to_reach)
        actor_velocity = KeepVelocity(self.other_actors[0], self._other_actor_target_velocity)
        actor_drive = DriveDistance(self.other_actors[0], 0.3*lane_width)
        actor_start_cross_lane = AccelerateToVelocity(self.other_actors[0], 1.0,
                                                      self._other_actor_target_velocity)
        actor_cross_lane = DriveDistance(self.other_actors[0], lane_width)
        actor_stop_crossed_lane = StopVehicle(self.other_actors[0], self._other_actor_max_brake)
        timeout_other_actor = TimeOut(10)
        actor_remove = ActorDestroy(self.other_actors[0])
        static_remove = ActorDestroy(self.other_actors[1])
        end_condition = DriveDistance(self.ego_vehicle, 50)

        # non leaf nodes
        root = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        scenario_sequence = py_trees.composites.Sequence()
        keep_velocity_other = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        keep_velocity = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        # building tree
        root.add_child(scenario_sequence)
        scenario_sequence.add_child(HandBrakeVehicle(self.other_actors[0], True))
        scenario_sequence.add_child(start_condition)
        scenario_sequence.add_child(HandBrakeVehicle(self.other_actors[0], False))
        scenario_sequence.add_child(keep_velocity)
        scenario_sequence.add_child(keep_velocity_other)
        scenario_sequence.add_child(actor_stop_crossed_lane)
        scenario_sequence.add_child(timeout_other_actor)
        scenario_sequence.add_child(actor_remove)
        scenario_sequence.add_child(static_remove)
        scenario_sequence.add_child(end_condition)

        keep_velocity.add_child(actor_velocity)
        keep_velocity.add_child(actor_drive)
        keep_velocity_other.add_child(actor_start_cross_lane)
        keep_velocity_other.add_child(actor_cross_lane)
        keep_velocity_other.add_child(TimeOut(5))

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
