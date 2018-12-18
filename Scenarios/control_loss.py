#!/usr/bin/env python

#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Control Loss Vehicle scenario:

The scenario realizes that the vehicle looses control due to
bad road conditions, etc. and checks to see if the vehicle
regains control and corrects it's course.
"""

import random

import py_trees
import carla

from ScenarioManager.atomic_scenario_behavior import *
from ScenarioManager.atomic_scenario_criteria import *
from ScenarioManager.scenario_manager import Scenario
from ScenarioManager.timer import TimeOut
from Scenarios.basic_scenario import *


class ControlLoss(BasicScenario):

    """
    Implementation of "Control Loss Vehicle" (Traffic Scenario 01)

    Location : Town02
    """

    timeout = 60            # Timeout of scenario in seconds

    # ego vehicle parameters
    _ego_vehicle_model = 'vehicle.carlamotors.carlacola'
    _ego_vehicle_start = carla.Transform(
        carla.Location(x=60, y=109.5, z=2.0), carla.Rotation(yaw=0))
    _no_of_jitter_actions = 20
    _ego_vehicle_driven_distance = 35
    _noise_mean = 0     # Mean value of steering noise
    _noise_std = 0.3    # Std. deviation of steerning noise

    def __init__(self, world, debug_mode=False):
        """
        Setup all relevant parameters and create scenario
        and instantiate scenario manager
        """
        self.ego_vehicle = setup_vehicle(world,
                                         self._ego_vehicle_model,
                                         self._ego_vehicle_start,
                                         hero=True)

        super(ControlLoss, self).__init__(name="ControlLoss",
                                          debug_mode=debug_mode)

    def _create_behavior(self):
        """
        The scenario defined after is a "control loss vehicle" scenario. After
        invoking this scenario, it will wait for the user controlled vehicle to
        enter the start region, then it performs a jitter action. Finally,
        the user-controlled vehicle has to reach a target region.
        If this does not happen within 60 seconds, a timeout stops the scenario
        """

        # start condition
        startcondition = InTriggerRegion(self.ego_vehicle, 75, 80, 100, 110)

        # jitter sequence
        jitterSequence = py_trees.composites.Sequence(
            "Jitter Sequence Behavior")
        jitterTimeout = TimeOut(timeout=0.2, name="Timeout for next jitter")

        for i in range(self._no_of_jitter_actions):
            ego_vehicle_max_steer = random.gauss(self._noise_mean, self._noise_std)

            # turn vehicle
            turn = SteerVehicle(
                self.ego_vehicle,
                ego_vehicle_max_steer,
                name='Steering ' + str(i))

            jitterAction = py_trees.composites.Parallel(
                "Jitter Actions with Timeouts",
                policy=py_trees.common.
                ParallelPolicy.SUCCESS_ON_ALL)
            jitterAction.add_child(turn)
            jitterAction.add_child(jitterTimeout)
            jitterSequence.add_child(jitterAction)

        target_location = carla.Location(x=150, y=112, z=2.0)

        # endcondition
        endcondition = InTriggerDistanceToLocation(self.ego_vehicle, target_location, 10,
                                                   "InDistanceLocation")

        # Build behavior tree
        sequence = py_trees.composites.Sequence("Sequence Behavior")
        sequence.add_child(startcondition)
        sequence.add_child(jitterSequence)
        sequence.add_child(endcondition)
        return sequence

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []

        collision_criterion = CollisionTest(self.ego_vehicle)
        # Region check to verify if the vehicle reached correct lane
        reached_region_criterion = ReachedRegionTest(
            self.ego_vehicle,
            115, 120,
            109, 112)

        criteria.append(collision_criterion)
        criteria.append(reached_region_criterion)

        return criteria
