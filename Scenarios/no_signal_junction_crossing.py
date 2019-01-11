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
import carla

from ScenarioManager.atomic_scenario_behavior import *
from ScenarioManager.atomic_scenario_criteria import *
from ScenarioManager.scenario_manager import Scenario
from ScenarioManager.timer import TimeOut
from Scenarios.basic_scenario import *


class NoSignalJunctionCrossing(BasicScenario):

    """
    Implementation class for
    'Non-signalized junctions: crossing negotiation' scenario,
    Traffic Scenario 10.

    Location    :   Town03
    """

    # ego vehicle parameters
    _ego_vehicle_model = 'vehicle.lincoln.mkz2017'
    _ego_vehicle_start = carla.Transform(
        carla.Location(x=-74.32, y=-50, z=0.5), carla.Rotation(yaw=270))
    _ego_vehicle_max_velocity = 20
    _ego_vehicle_driven_distance = 105

    # other vehicle
    _other_vehicle_model = 'vehicle.tesla.model3'
    _other_vehicle_start = carla.Transform(
        carla.Location(x=-105, y=-136, z=0.5), carla.Rotation(yaw=0))
    _other_vehicle_max_brake = 1.0
    _other_vehicle_target_velocity = 15

    def __init__(self, world, debug_mode=False):
        """
        Setup all relevant parameters and create scenario
        and instantiate scenario manager
        """
        self.other_vehicles = [setup_vehicle(
            world,
            self._other_vehicle_model,
            self._other_vehicle_start)]
        self.ego_vehicle = setup_vehicle(
            world,
            self._ego_vehicle_model,
            self._ego_vehicle_start, hero=True)
        super(NoSignalJunctionCrossing, self).__init__(
            name="NoSignalJunctionCrossing",
            town="Town03",
            world=world,
            debug_mode=debug_mode)

    def _create_behavior(self):
        """
        After invoking this scenario, it will wait for the user
        controlled vehicle to enter the start region,
        then make a traffic participant to accelerate
        until it is going fast enough to reach an intersection point.
        at the same time as the user controlled vehicle at the junction.
        Once the user controlled vehicle comes close to the junction,
        the traffic participant accelerates and passes through the junction.
        After 60 seconds, a timeout stops the scenario.
        """

        # Creating leaf nodes
        start_other_trigger = InTriggerRegion(
            self.ego_vehicle,
            -80, -70,
            -75, -60)

        sync_arrival = SyncArrival(
            self.other_vehicles[0], self.ego_vehicle,
            carla.Location(x=-74.63, y=-136.34))

        pass_through_trigger = InTriggerRegion(
            self.ego_vehicle,
            -90, -70,
            -124, -119)

        keep_velocity_other = KeepVelocity(
            self.other_vehicles[0],
            self._other_vehicle_target_velocity)

        stop_other_trigger = InTriggerRegion(
            self.other_vehicles[0],
            -45, -35,
            -140, -130)

        stop_other = StopVehicle(
            self.other_vehicles[0],
            self._other_vehicle_max_brake)

        end_condition = InTriggerRegion(
            self.ego_vehicle,
            -90, -70,
            -170, -156
        )

        # Creating non-leaf nodes
        root = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        scenario_sequence = py_trees.composites.Sequence()
        sync_arrival_parallel = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        keep_velocity_other_parallel = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        # Building tree
        root.add_child(scenario_sequence)
        scenario_sequence.add_child(start_other_trigger)
        scenario_sequence.add_child(sync_arrival_parallel)
        scenario_sequence.add_child(keep_velocity_other_parallel)
        scenario_sequence.add_child(stop_other)
        scenario_sequence.add_child(end_condition)
        sync_arrival_parallel.add_child(sync_arrival)
        sync_arrival_parallel.add_child(pass_through_trigger)
        keep_velocity_other_parallel.add_child(keep_velocity_other)
        keep_velocity_other_parallel.add_child(stop_other_trigger)

        return root

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []

        # Adding checks for ego vehicle
        collision_criterion_ego = CollisionTest(self.ego_vehicle)
        driven_distance_criterion = DrivenDistanceTest(
            self.ego_vehicle, self._ego_vehicle_driven_distance)
        criteria.append(collision_criterion_ego)
        criteria.append(driven_distance_criterion)

        # Add approriate checks for other vehicles
        for vehicle in self.other_vehicles:
            collision_criterion = CollisionTest(vehicle)
            criteria.append(collision_criterion)

        return criteria
