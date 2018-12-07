#!/usr/bin/env python

#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Control Loss Vehicle scenario:

The scenario realizes that the vehicle looses control due to
bad road conditions, etc.
"""

import random
import sys

import py_trees
import carla

from ScenarioManager.atomic_scenario_behavior import *
from ScenarioManager.atomic_scenario_criteria import *
from ScenarioManager.scenario_manager import Scenario
from ScenarioManager.timer import TimeOut
from Scenarios.basic_scenario import *


class ControlLoss(BasicScenario):

    """
    This class holds everything required for a simple "Control Loss Vehicle"
    to perform jitter sequence with steering angle.
    """

    timeout = 60            # Timeout of scenario in seconds

    # ego vehicle parameters
    ego_vehicle_model = 'vehicle.carlamotors.carlacola'
    ego_vehicle_start = carla.Transform(
        carla.Location(x=60, y=109.5, z=2.0), carla.Rotation(yaw=0))
    ego_vehicle_max_steer = 0.1
    no_of_jitter_actions = 5
    ego_vehicle_driven_distance = 35

    # other vehicle parameter
    other_vehicles = []

    def __init__(self, world, debug_mode=False):
        """
        Setup all relevant parameters and create scenario
        and instantiate scenario manager
        """
        self.ego_vehicle = setup_vehicle(world,
                                         self.ego_vehicle_model,
                                         self.ego_vehicle_start,
                                         hero=True)

        super(ControlLoss, self).__init__(name="ControlLoss",
                                          debug_mode=debug_mode)

    def create_behavior(self):
        """
        The scenario defined after is a "control loss vehicle" scenario. After
        invoking this scenario, it will wait for the user controlled vehicle to
        enter the start region, then it performs a jitter action. Finally, the user-controlled
        vehicle has to reach a target region.
        If this does not happen within 60 seconds, a timeout stops the scenario
        """

        # start condition
        startcondition = InTriggerRegion(self.ego_vehicle, 75, 80, 100, 110)

        # jitter sequence
        jitterSequence = py_trees.composites.Sequence(
            "Jitter Sequence Behavior")
        jitterTimeout = TimeOut(timeout=1.0, name="Timeout for next jitter")

        for i in range(self.no_of_jitter_actions):
            jitter_steer = self.ego_vehicle_max_steer
            if i % 2 != 0:
                jitter_steer = self.ego_vehicle_max_steer * -1.0
            # turn vehicle
            turn = SteerVehicle(
                self.ego_vehicle,
                jitter_steer,
                name='Steering ' + str(i))
            jitterAction = py_trees.composites.Parallel("Jitter Actions with Timeouts",
                                                        policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)
            jitterAction.add_child(turn)
            if i == 0:
                jitterAction.add_child(TimeOut(0.5))
            else:
                jitterAction.add_child(jitterTimeout)
            jitterSequence.add_child(jitterAction)

        # Build behavior tree
        sequence = py_trees.composites.Sequence("Sequence Behavior")
        sequence.add_child(startcondition)
        sequence.add_child(jitterSequence)
        sequence.add_child(TimeOut(20))
        return sequence

    def create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []

        collision_criterion = CollisionTest(self.ego_vehicle)
        driven_distance_criterion = DrivenDistanceTest(
            self.ego_vehicle,
            self.ego_vehicle_driven_distance)
        reached_region_criterion = ReachedRegionTest(
            self.ego_vehicle,
            115, 120,
            104, 110)
        keep_lane_criterion = KeepLaneTest(self.ego_vehicle)

        criteria.append(collision_criterion)
        criteria.append(driven_distance_criterion)
        criteria.append(reached_region_criterion)
        criteria.append(keep_lane_criterion)

        return criteria
