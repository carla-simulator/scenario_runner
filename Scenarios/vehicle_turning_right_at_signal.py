#!/usr/bin/env python

#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Non-signalized junctions: crossing negotiation:

The hero vehicle is passing through a junction without traffic lights
And encounters another vehicle passing across the junction.
"""

import py_trees

from ScenarioManager.atomic_scenario_behavior import *
from ScenarioManager.atomic_scenario_criteria import *
from ScenarioManager.timer import TimeOut
from Scenarios.basic_scenario import *

VEHICLE_TURNING_SIGNAL_SCENARIOS = ["VehicleTurningRightAtSignal"]

class VehicleTurningRightAtSignal(BasicScenario):

    """
    Implementation class for
    'Vehicle turning right at signalized junction' scenario,
    Traffic Scenario 09.
    """

    def __init__(self, world, ego_vehicle, other_actors, town, randomize=False, debug_mode=False):
        """
        Setup all relevant parameters and create scenario
        and instantiate scenario manager
        """

        super(VehicleTurningRightAtSignal, self).__init__(
            "VehicleTurningRightAtSignal",
            ego_vehicle,
            other_actors,
            town,
            world,
            debug_mode)

        if debug_mode:
            py_trees.logging.level = py_trees.logging.Level.DEBUG


    def _create_behavior(self):
        """
        The ego vehicle is passing through a junction and a traffic participant
        takes a right turn on to the ego vehicle's lane. The ego vehicle has to
        navigate the scenario without collision with the participant and cross
        the junction.
        """

        # Creating leaf nodes
        start_trigger_location, _ = get_location_in_distance(self.ego_vehicle, 1)
        start_other_trigger = InTriggerDistanceToLocation(
            self.ego_vehicle, start_trigger_location, 2.0)
        sync_arrival = SyncArrival(self.other_actors[0], self.ego_vehicle, carla.Location(x=85, y=-136))
        pass_through_trigger = InTriggerDistanceToNextIntersection(self.other_actors[0], 1)
        keep_velocity_other = KeepVelocity(self.other_actors[0], 15)
        other_end_trigger = DriveDistance(self.other_actors[0], 10)
        end_trigger_location, _ = get_location_in_distance(self.ego_vehicle, 100)
        end_condition = InTriggerDistanceToLocation(
            self.ego_vehicle, end_trigger_location, 10.0)

        root_timeout = TimeOut(self.timeout)

        # Creating non-leaf nodes
        root = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        scenario_sequence = py_trees.composites.Sequence()
        sync_arrival_parallel = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        keep_velocity_parallel = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        # Building tree
        root.add_child(scenario_sequence)
        root.add_child(root_timeout)
        scenario_sequence.add_child(start_other_trigger)
        scenario_sequence.add_child(sync_arrival_parallel)
        scenario_sequence.add_child(keep_velocity_parallel)
        scenario_sequence.add_child(end_condition)
        sync_arrival_parallel.add_child(sync_arrival)
        sync_arrival_parallel.add_child(pass_through_trigger)
        keep_velocity_parallel.add_child(keep_velocity_other)
        keep_velocity_parallel.add_child(other_end_trigger)

        return root

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []

        # Adding checks for ego vehicle
        collision_criterion_ego = CollisionTest(self.ego_vehicle)
        criteria.append(collision_criterion_ego)

        # Add approriate checks for other vehicles
        for vehicle in self.other_actors:
            collision_criterion = CollisionTest(vehicle)
            criteria.append(collision_criterion)

        return criteria
