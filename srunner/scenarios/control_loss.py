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

from srunner.scenariomanager.atomic_scenario_behavior import *
from srunner.scenariomanager.atomic_scenario_criteria import *
from srunner.scenariomanager.timer import TimeOut
from srunner.scenarios.basic_scenario import *
from srunner.scenarios.scenario_helper import *

CONTROL_LOSS_SCENARIOS = [
    "ControlLoss"
]


class ControlLoss(BasicScenario):

    """
    Implementation of "Control Loss Vehicle" (Traffic Scenario 01)
    """

    category = "ControlLoss"

    timeout = 60            # Timeout of scenario in seconds

    # ego vehicle parameters
    _no_of_jitter_actions = 20
    _noise_mean = 0      # Mean value of steering noise
    _noise_std = 0.02    # Std. deviation of steering noise
    _dynamic_mean_for_steer = 0.01
    _dynamic_mean_for_throttle = 0.75
    _abort_distance_to_intersection = 10
    _start_distance = 20
    _end_distance = 80
    _ego_vehicle_max_steer = 0.0
    _ego_vehicle_max_throttle = 1.0
    _ego_vehicle_target_velocity = 15

    def __init__(self, world, ego_vehicle, config, randomize=False, debug_mode=False):
        """
        Setup all relevant parameters and create scenario
        """
        super(ControlLoss, self).__init__("ControlLoss",
                                          ego_vehicle,
                                          config,
                                          world,
                                          debug_mode)

    def _create_behavior(self):
        """
        The scenario defined after is a "control loss vehicle" scenario. After
        invoking this scenario, it will wait until the vehicle drove a few meters
        (_start_distance), and then perform a jitter action. Finally, the vehicle
        has to reach a target point (_end_distance). If this does not happen within
        60 seconds, a timeout stops the scenario
        """

        # start condition
        location, _ = get_location_in_distance(self.ego_vehicle, self._start_distance)
        start_condition = InTriggerDistanceToLocation(self.ego_vehicle, location, 10.0)

        # jitter sequence
        jitter_sequence = py_trees.composites.Sequence("Jitter Sequence Behavior")
        jitter_timeout = TimeOut(timeout=0.2, name="Timeout for next jitter")

        for i in range(self._no_of_jitter_actions):
            noise = random.gauss(self._noise_mean, self._noise_std)
            noise = abs(noise)
            self._ego_vehicle_max_steer = min(0, -(noise - self._dynamic_mean_for_steer))
            self._ego_vehicle_max_throttle = min(noise + self._dynamic_mean_for_throttle, 1)

            # turn vehicle
            turn = AddNoiseToVehicle(
                self.ego_vehicle, self._ego_vehicle_max_steer, self._ego_vehicle_max_throttle, name='Jittering ' + str(i))

            jitter_action = py_trees.composites.Parallel("Jitter Actions with Timeouts",
                                                         policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)
            jitter_action.add_child(turn)
            jitter_action.add_child(jitter_timeout)
            jitter_sequence.add_child(jitter_action)

        # Abort jitter_sequence, if the vehicle is approaching an intersection
        jitter_abort = InTriggerDistanceToNextIntersection(self.ego_vehicle, self._abort_distance_to_intersection)

        jitter = py_trees.composites.Parallel("Jitter",
                                              policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        jitter.add_child(jitter_sequence)
        jitter.add_child(jitter_abort)

        # endcondition: Check if vehicle reached waypoint _end_distance from here:
        end_condition = DriveDistance(self.ego_vehicle, self._end_distance)

        # Build behavior tree
        sequence = py_trees.composites.Sequence("Sequence Behavior")
        sequence.add_child(start_condition)
        sequence.add_child(jitter)
        sequence.add_child(end_condition)
        return sequence

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []

        collision_criterion = CollisionTest(self.ego_vehicle)
        criteria.append(collision_criterion)

        return criteria
