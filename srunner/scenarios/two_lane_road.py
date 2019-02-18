#!/usr/bin/env python

# Copyright (c) 2019 Aptiv.
# authors: Tomasz Sulkowski (tomasz.sulkowski@aptiv.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Scenarios in which ego car meets other vehicles on a two lane road. Made for
benchmarking sensors as well as behavior tests.

Those include passing an oncoming vehicle, overtaking slow vehicles, following
accelerating and decelerating vehicles, avoiding collision of drifting oncoming
and parallel driving vehicles as well as merging from driveway.
"""

import py_trees
import carla

from srunner.scenariomanager.extended_scenario_behavior import *
from srunner.scenariomanager.atomic_scenario_behavior import *
from srunner.scenariomanager.atomic_scenario_criteria import *
from srunner.scenarios.basic_scenario import *

TWO_LANE_ROAD_SCENARIOS = [
    "PassingFromOppositeDirections",
    "OvertakingSlowTarget",
    "FollowingAcceleratingTarget",
    "FollowingDeceleratingTarget",
    "FollowingChangingLanesTarget",
    "DrivingOffDriveway",
    "OncomingTargetDriftsOntoEgoLane"
]


def kmh_to_ms(kmh):
    """
    Converts km/h to m/s
    """
    return kmh/3.6


class PassingFromOppositeDirections(BasicScenario):

    """
    This class holds everything required for a scenario,
    in which other vehicle is passing ego from opposite direction
    """

    category = "TwoLaneRoad"
    timeout = 120

    VehicleModelToSpeed = {
        'vehicle.tesla.model3': kmh_to_ms(45),
        'vehicle.carlamotors.carlacola': kmh_to_ms(45),
        'vehicle.yamaha.yzf': kmh_to_ms(45),
        'vehicle.gazelle.omafiets': kmh_to_ms(15)
    }

    def __init__(self, world, ego_vehicle, other_actors, town, randomize=False, debug_mode=False):
        """
        Setup other vehicle target speed based on its type
        for typical urban scenario
        """

        self.target_speed = self.VehicleModelToSpeed[other_actors[0].type_id]

        super(PassingFromOppositeDirections, self).__init__(
            name="PassingFromOppositeDirections",
            ego_vehicle=ego_vehicle,
            other_actors=other_actors,
            town=town,
            world=world,
            debug_mode=debug_mode)

    def _create_behavior(self):
        """
        Scenario behavior:
        The other vehicle drives on opposite lane. Ego should pass it and drive
        safely untill the end of straight.

        If this does not happen within 120 seconds, a timeout stops the scenario
        """

        sequence = py_trees.composites.Sequence("Main Sequence")

        parallel = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        parallel.add_child(DriveToLocationContinuous(
            self.other_actors[0], carla.Location(x=396, y=8), max_speed=self.target_speed))
        parallel.add_child(TriggerOnLocation(self.ego_vehicle, 392, 315, 5))
        sequence.add_child(parallel)

        return sequence

    def _create_test_criteria(self):
        """
        Creates a list of criteria: ego car should not crash
        and reach end of straight
        """
        criteria = []

        criteria.append(CollisionTest(self.ego_vehicle))
        criteria.append(ReachedRegionTest(self.ego_vehicle, 390, 400, 310, 320))

        return criteria


class OvertakingSlowTarget(BasicScenario):

    """
    This class holds everything required for a scenario,
    in which ego car should overtake other vehicle safely
    """

    category = "TwoLaneRoad"
    timeout = 90

    VehicleModelToSpeedDistance = {
        'vehicle.tesla.model3': [kmh_to_ms(25), 0.5],
        'vehicle.carlamotors.carlacola': [kmh_to_ms(25), 0.5],
        'vehicle.yamaha.yzf': [kmh_to_ms(25), 1],
        'vehicle.gazelle.omafiets': [kmh_to_ms(15), 1.5]
    }

    def __init__(self, world, ego_vehicle, other_actors, town, randomize=False, debug_mode=False):
        """
        Setup other vehicle target speed based on its type
        as well as minimal lateral distance to be classified as safe
        for typical urban scenario
        """

        self.target_speed = self.VehicleModelToSpeedDistance[other_actors[0].type_id][0]
        self.min_lateral_distance = self.VehicleModelToSpeedDistance[other_actors[0].type_id][1]

        self.ego_veh_width = ego_vehicle.bounding_box.extent.y*2
        self.target_veh_width = other_actors[0].bounding_box.extent.y*2

        super(OvertakingSlowTarget, self).__init__(
            name="OvertakingSlowTarget",
            ego_vehicle=ego_vehicle,
            other_actors=other_actors,
            town=town,
            world=world,
            debug_mode=debug_mode)

    def _create_behavior(self):
        """
        Scenario behavior:
        The other vehicle drives on the same lane and direction in front of ego.
        Ego should overtake it using opposite lane with minimum safe lateral
        distance for a typical urban scenario.

        If this does not happen within 90 seconds, a timeout stops the scenario
        """

        sequence = py_trees.composites.Sequence("Main Sequence")

        # Check whether the ego had at least self.min_lateral_distance meters
        # between itself and Target during overtake - otherwise the scenario
        # won't continue and time out
        parallel = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        parallel.add_child(DriveToLocationContinuous(self.other_actors[0], carla.Location(x=392, y=300),
                                                     max_speed=self.target_speed))
        parallel.add_child(TriggerOnStatusChange(
            self.ego_vehicle, x=392+self.ego_veh_width/2+self.min_lateral_distance+self.target_veh_width/2))
        sequence.add_child(parallel)

        # if target reaches end of straight, end scenario
        parallel = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        parallel.add_child(DriveToLocationContinuous(self.other_actors[0], carla.Location(x=392, y=300),
                                                     max_speed=self.target_speed))
        parallel.add_child(TriggerOnLocation(self.other_actors[0], 392, 300, 1))
        sequence.add_child(parallel)

        return sequence

    def _create_test_criteria(self):
        """
        Creates a list of criteria: ego car should not crash
        and both vehicles should reach end of straight
        """
        criteria = []

        criteria.append(CollisionTest(self.ego_vehicle))
        # Ego should reach region in front of target
        criteria.append(ReachedRegionTest(self.ego_vehicle, 390, 394, 302, 320))
        criteria.append(ReachedRegionTest(self.other_actors[0], 390, 400, 299, 301))

        return criteria


class FollowingAcceleratingTarget(BasicScenario):

    """
    This class holds everything required for a scenario,
    in which ego car should follow slowly accelerating other vehicle
    """

    category = "TwoLaneRoad"
    timeout = 90

    VehicleModelToSpeed = {
        'vehicle.tesla.model3': kmh_to_ms(40),
        'vehicle.carlamotors.carlacola': kmh_to_ms(40),
        'vehicle.yamaha.yzf': kmh_to_ms(40),
        'vehicle.gazelle.omafiets': kmh_to_ms(20)
    }

    def __init__(self, world, ego_vehicle, other_actors, town, randomize=False, debug_mode=False):
        """
        Setup other vehicle target speed based on its type
        for typical urban scenario
        """

        self.target_speed = self.VehicleModelToSpeed[other_actors[0].type_id]

        super(FollowingAcceleratingTarget, self).__init__(
            name="FollowingAcceleratingTarget",
            ego_vehicle=ego_vehicle,
            other_actors=other_actors,
            town=town,
            world=world,
            debug_mode=debug_mode)

    def _create_behavior(self):
        """
        Scenario behavior:
        The other vehicle drives on the same lane and direction in front of ego.
        When approached by ego it starts to slowly accelerate until target speed
        is reached.

        If this does not happen within 90 seconds, a timeout stops the scenario
        """
        sequence = py_trees.composites.Sequence("Main Sequence")

        parallel = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        parallel.add_child(FollowVehicleContinuous(
            self.other_actors[2], self.other_actors[1], max_speed=kmh_to_ms(30)))
        parallel.add_child(FollowVehicleContinuous(
            self.other_actors[3], self.other_actors[2], max_speed=kmh_to_ms(20), max_throttle=0.9))
        parallel.add_child(FollowVehicleContinuous(
            self.other_actors[4], self.other_actors[3], max_speed=kmh_to_ms(12), max_throttle=0.8))
        parallel.add_child(FollowVehicleContinuous(
            self.other_actors[5], self.other_actors[4], max_speed=kmh_to_ms(7), max_throttle=0.7))
        parallel.add_child(InTriggerDistanceToVehicle(self.ego_vehicle, self.other_actors[0], 20))
        sequence.add_child(parallel)

        parallel = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        parallel.add_child(FollowVehicleContinuous(
            self.other_actors[2], self.other_actors[1], max_speed=kmh_to_ms(30)))
        parallel.add_child(FollowVehicleContinuous(
            self.other_actors[3], self.other_actors[2], max_speed=kmh_to_ms(20), max_throttle=0.9))
        parallel.add_child(FollowVehicleContinuous(
            self.other_actors[4], self.other_actors[3], max_speed=kmh_to_ms(12), max_throttle=0.8))
        parallel.add_child(FollowVehicleContinuous(
            self.other_actors[5], self.other_actors[4], max_speed=kmh_to_ms(7), max_throttle=0.7))
        parallel.add_child(DriveToLocationContinuous(
            self.other_actors[0], carla.Location(x=392, y=308), max_throttle=0.75, max_speed=self.target_speed/1.5))
        parallel.add_child(TriggerOnLocation(self.other_actors[0], 392, 200, 1))
        sequence.add_child(parallel)

        parallel = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        parallel.add_child(UseAutoPilot(self.other_actors[2]))
        parallel.add_child(UseAutoPilot(self.other_actors[3]))
        parallel.add_child(UseAutoPilot(self.other_actors[4]))
        parallel.add_child(UseAutoPilot(self.other_actors[5]))
        parallel.add_child(DriveToLocationContinuous(
            self.other_actors[0], carla.Location(x=392, y=308), max_throttle=1, max_speed=self.target_speed))
        parallel.add_child(TriggerOnLocation(self.other_actors[0], 392, 300, 1))
        sequence.add_child(parallel)

        return sequence

    def _create_test_criteria(self):
        """
        Creates a list of criteria: ego car should not crash
        """
        criteria = []

        criteria.append(CollisionTest(self.ego_vehicle))

        return criteria


class FollowingDeceleratingTarget(BasicScenario):

    """
    This class holds everything required for a scenario,
    in which ego car should follow slowly decelerating other vehicle
    """

    category = "TwoLaneRoad"
    timeout = 90

    VehicleModelToSpeed = {
        'vehicle.tesla.model3': kmh_to_ms(40),
        'vehicle.carlamotors.carlacola': kmh_to_ms(40),
        'vehicle.yamaha.yzf': kmh_to_ms(40),
        'vehicle.gazelle.omafiets': kmh_to_ms(20)
    }

    def __init__(self, world, ego_vehicle, other_actors, town, randomize=False, debug_mode=False):
        """
        Setup other vehicle target speed based on its type
        for typical urban scenario
        """
        self.target_speed = self.VehicleModelToSpeed[other_actors[0].type_id]

        super(FollowingDeceleratingTarget, self).__init__(
            name="FollowingDeceleratingTarget",
            ego_vehicle=ego_vehicle,
            other_actors=other_actors,
            town=town,
            world=world,
            debug_mode=debug_mode)

    def _create_behavior(self):
        """
        Scenario behavior:
        The other vehicle drives on the same lane and direction in front of ego.
        When approached by ego it accelerates quickly to target speed,
        next starts to slowly decelerate until stopped.

        If this does not happen within 90 seconds, a timeout stops the scenario
        """
        sequence = py_trees.composites.Sequence("Main Sequence")

        parallel = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        parallel.add_child(FollowVehicleContinuous(
            self.other_actors[2], self.other_actors[1], max_speed=kmh_to_ms(30)))
        parallel.add_child(FollowVehicleContinuous(
            self.other_actors[3], self.other_actors[2], max_speed=kmh_to_ms(25), max_throttle=0.9))
        parallel.add_child(FollowVehicleContinuous(
            self.other_actors[4], self.other_actors[3], max_speed=kmh_to_ms(20), max_throttle=0.8))
        parallel.add_child(FollowVehicleContinuous(
            self.other_actors[5], self.other_actors[4], max_speed=kmh_to_ms(15), max_throttle=0.7))
        parallel.add_child(InTriggerDistanceToVehicle(self.ego_vehicle, self.other_actors[0], 50))
        sequence.add_child(parallel)

        parallel = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        parallel.add_child(UseAutoPilot(self.other_actors[2]))
        parallel.add_child(UseAutoPilot(self.other_actors[3]))
        parallel.add_child(UseAutoPilot(self.other_actors[4]))
        parallel.add_child(UseAutoPilot(self.other_actors[5]))
        parallel.add_child(DriveToLocationContinuous(
            self.other_actors[0], carla.Location(x=392, y=308), max_speed=self.target_speed))
        parallel.add_child(TriggerOnLocation(self.other_actors[0], 392, 150, 1))
        sequence.add_child(parallel)

        parallel = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        parallel.add_child(DriveToLocationContinuous(
            self.other_actors[0], carla.Location(x=392, y=308), max_braking=0.01, max_throttle=0.5,
            max_speed=self.target_speed/1.5))
        parallel.add_child(TriggerOnLocation(self.other_actors[0], 392, 170, 1))
        sequence.add_child(parallel)

        parallel = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        parallel.add_child(DriveToLocationContinuous(
            self.other_actors[0], carla.Location(x=392, y=308), max_braking=0.01, max_throttle=0.4,
            max_speed=self.target_speed/2))
        parallel.add_child(TriggerOnLocation(self.other_actors[0], 392, 180, 1))
        sequence.add_child(parallel)

        parallel = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        parallel.add_child(DriveToLocationContinuous(
            self.other_actors[0], carla.Location(x=392, y=308), max_braking=0.01, max_throttle=0.3,
            max_speed=self.target_speed/4))
        parallel.add_child(TriggerOnLocation(self.other_actors[0], 392, 185, 1))
        sequence.add_child(parallel)

        sequence.add_child(StopVehicle(self.other_actors[0], 0.1))

        return sequence

    def _create_test_criteria(self):
        """
        Creates a list of criteria: ego car should not crash
        """
        criteria = []

        criteria.append(CollisionTest(self.ego_vehicle))

        return criteria


class FollowingChangingLanesTarget(BasicScenario):

    """
    This class holds everything required for a scenario,
    in which ego car approaches other vehicle slowly changing lane to one
    occupied by ego (cut off)
    """

    category = "TwoLaneRoad"
    timeout = 90

    VehicleModelToSpeed = {
        'vehicle.tesla.model3': kmh_to_ms(40),
        'vehicle.carlamotors.carlacola': kmh_to_ms(40),
        'vehicle.yamaha.yzf': kmh_to_ms(40),
        'vehicle.gazelle.omafiets': kmh_to_ms(20)
    }

    def __init__(self, world, ego_vehicle, other_actors, town, randomize=False, debug_mode=False):
        """
        Setup other vehicle target speed based on its type
        for typical urban scenario
        """
        self.ego_veh_width = ego_vehicle.bounding_box.extent.y*2
        self.target_speed = self.VehicleModelToSpeed[other_actors[0].type_id]

        super(FollowingChangingLanesTarget, self).__init__(
            name="FollowingChangingLanesTarget",
            ego_vehicle=ego_vehicle,
            other_actors=other_actors,
            town=town,
            world=world,
            debug_mode=debug_mode)

    def _create_behavior(self):
        """
        Scenario behavior:
        The other vehicle drives in the same direction in front of ego.
        When approached by ego it mathes its speed and slowly drives onto ego
        lane.
        Once the change is done it continues driving till the end of straight.

        If this does not happen within 90 seconds, a timeout stops the scenario
        """
        sequence = py_trees.composites.Sequence("Main Sequence")

        parallel = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        parallel.add_child(DriveToLocationContinuous(
            self.other_actors[0], carla.Location(x=389, y=158), max_speed=kmh_to_ms(5)))
        parallel.add_child(TriggerOnStatusChange(self.other_actors[0], y=62))
        sequence.add_child(parallel)

        parallel = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        parallel.add_child(SyncArrival(self.other_actors[0], self.ego_vehicle, carla.Location(x=392, y=158, z=38.5)))
        parallel.add_child(TriggerOnStatusChange(self.other_actors[0], x=392+self.ego_veh_width/2))
        parallel.add_child(InTriggerDistanceToVehicle(self.ego_vehicle, self.other_actors[0], 10))
        sequence.add_child(parallel)

        parallel = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        parallel.add_child(DriveToLocationContinuous(
            self.other_actors[0], carla.Location(x=392, y=308), max_braking=0.01, max_speed=self.target_speed))
        parallel.add_child(TriggerOnLocation(self.other_actors[0], 392, 208, 3))
        sequence.add_child(parallel)

        return sequence

    def _create_test_criteria(self):
        """
        Creates a list of criteria: ego car should not crash
        """
        criteria = []

        criteria.append(CollisionTest(self.ego_vehicle))

        return criteria


class DrivingOffDriveway(BasicScenario):

    """
    This class holds everything required for a scenario,
    in which ego car meets other vehicle driving off an covered driveway right
    in front of ego.
    """

    category = "TwoLaneRoad"
    timeout = 60

    def __init__(self, world, ego_vehicle, other_actors, town, randomize=False, debug_mode=False):
        """
        Setup scenario
        """
        super(DrivingOffDriveway, self).__init__(
            name="DrivingOffDriveway",
            ego_vehicle=ego_vehicle,
            other_actors=other_actors,
            town=town,
            world=world,
            debug_mode=debug_mode)

    def _create_behavior(self):
        """
        Scenario behavior:
        Other vehicle will match ego car position while driving out of driveway
        so that it will roll out right in front of ego.
        After a brief stop it will turn right and continue driving on ego lane.

        If this does not happen within 60 seconds, a timeout stops the scenario
        """
        sequence = py_trees.composites.Sequence("Main Sequence")

        parallel = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        parallel.add_child(SyncArrival(
            self.other_actors[0], self.ego_vehicle, carla.Location(x=268, y=59, z=38.5)))
        parallel.add_child(SyncArrival(
            self.other_actors[1], self.ego_vehicle, carla.Location(x=260, y=55, z=38.5)))
        parallel.add_child(TriggerOnLocation(self.other_actors[0], 271.5, 60, 2))
        sequence.add_child(parallel)

        sequence.add_child(StopVehicle(self.other_actors[0], 1))

        parallel = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        parallel.add_child(DriveToLocationContinuous(
            self.other_actors[0], carla.Location(x=290, y=60), max_speed=6, use_reverse=False))
        parallel.add_child(UseAutoPilot(self.other_actors[1]))
        parallel.add_child(TriggerOnLocation(self.other_actors[0], 290, 60, 2))
        sequence.add_child(parallel)

        parallel = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        parallel.add_child(UseAutoPilot(self.other_actors[0]))
        parallel.add_child(UseAutoPilot(self.other_actors[1]))
        parallel.add_child(TriggerOnLocation(self.ego_vehicle, 310, 60, 5))
        sequence.add_child(parallel)

        return sequence

    def _create_test_criteria(self):
        """
        Creates a list of criteria: ego car should not crash
        and reach end of straight.
        """
        criteria = []

        criteria.append(CollisionTest(self.ego_vehicle))
        criteria.append(ReachedRegionTest(self.ego_vehicle, 305, 315, 55, 65))

        return criteria


class OncomingTargetDriftsOntoEgoLane(BasicScenario):

    """
    This class holds everything required for a scenario,
    in which other vehicle is driving from opposite direction compared to ego
    and slowly drifts onto lane occupied by ego.
    Once the meeting point is reached or distance between vehicles is lower than
    30 meters, other vehicle veers off onto its proper lane.

    If this does not happen within 60 seconds, a timeout stops the scenario
    """

    category = "TwoLaneRoad"
    timeout = 60

    target_speed = kmh_to_ms(40)

    def __init__(self, world, ego_vehicle, other_actors, town, randomize=False, debug_mode=False):
        """
        Setup scenario by calculating meeting point using both car widths
        """
        self.ego_veh_width = ego_vehicle.bounding_box.extent.y*2
        self.target_veh_width = other_actors[0].bounding_box.extent.y*2
        self.destination_x = 392+self.ego_veh_width/4+self.target_veh_width/4

        super(OncomingTargetDriftsOntoEgoLane, self).__init__(
            name="OncomingTargetDriftsOntoEgoLane",
            ego_vehicle=ego_vehicle,
            other_actors=other_actors,
            town=town,
            world=world,
            debug_mode=debug_mode)

    def _create_behavior(self):
        """
        Scenario behavior:
        Other vehicle will turn slightly to point at the calculated meeting
        point. Next, it will match ego car position while drifting out of its
        lane onto oncoming ego.
        When the meeting point is within 20 meters or ego car is within 30
        meters, other vehicle will veer off onto its proper lane.

        Ego car should avoid other vehicle by slowing down and/or keep as close
        to right side of its lane as possible.

        If this does not happen within 60 seconds, a timeout stops the scenario
        """
        sequence = py_trees.composites.Sequence("Main Sequence")

        parallel = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        parallel.add_child(DriveToLocationContinuous(
            self.other_actors[0], carla.Location(x=self.destination_x, y=158)))
        parallel.add_child(TriggerOnStatusChange(self.other_actors[0], y=315))
        sequence.add_child(parallel)

        parallel = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        parallel.add_child(SyncArrival(
            self.other_actors[0], self.ego_vehicle, carla.Location(x=self.destination_x, y=158, z=38.5)))
        parallel.add_child(TriggerOnLocation(self.other_actors[0], self.destination_x, 158, 20))
        parallel.add_child(InTriggerDistanceToVehicle(self.ego_vehicle, self.other_actors[0], 30))
        sequence.add_child(parallel)

        parallel = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        parallel.add_child(DriveToLocationContinuous(
            self.other_actors[0], carla.Location(x=397, y=148), max_braking=0.01, max_speed=self.target_speed))
        parallel.add_child(TriggerOnLocation(self.other_actors[0], 397, 148, 3))
        sequence.add_child(parallel)

        parallel = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        parallel.add_child(DriveToLocationContinuous(
            self.other_actors[0], carla.Location(x=396, y=8), max_braking=0.01, max_speed=self.target_speed))
        parallel.add_child(TriggerOnLocation(self.ego_vehicle, 392, 258, 2))
        sequence.add_child(parallel)

        return sequence

    def _create_test_criteria(self):
        """
        Creates a list of criteria: ego car should not crash
        """
        criteria = []

        criteria.append(CollisionTest(self.ego_vehicle))

        return criteria
