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


class VehicleTurningRightAtSignal(BasicScenario):

    """
    Implementation class for
    'Vehicle turning right at signalized junction' scenario,
    Traffic Scenario 09.

    Location    :   Town03
    """

    # ego vehicle parameters
    _ego_vehicle_model = 'vehicle.carlamotors.carlacola'
    _ego_vehicle_start = carla.Transform(
        carla.Location(x=135, y=-136.5, z=10), carla.Rotation(yaw=180))

    # other vehicle
    _other_vehicle_model = 'vehicle.tesla.model3'
    _other_vehicle_start = carla.Transform(
        carla.Location(x=82.33, y=-160, z=8), carla.Rotation(yaw=90))

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
        super(VehicleTurningRightAtSignal, self).__init__(
            name="VehicleTurningRightAtSignal",
            town="Town03",
            world=world,
            debug_mode=debug_mode)

        # Setup scenario

        if debug_mode:
            py_trees.logging.level = py_trees.logging.Level.DEBUG

        behavior = self._create_behavior()
        criteria = self._create_test_criteria()
        self.scenario = Scenario(
            behavior, criteria, self.name, self.timeout)

    def _create_behavior(self):
        """
        The ego vehicle is passing through a junction and a traffic participant
        takes a right turn on to the ego vehicle's lane. The ego vehicle has to
        navigate the scenario without collision with the participant and cross
        the junction.
        """

        # Creating leaf nodes
        start_other_trigger = InTriggerRegion(
            self.ego_vehicle,
            130, 132,
            -140, -130)

        sync_arrival = SyncArrival(
            self.other_vehicles[0], self.ego_vehicle,
            carla.Location(x=81.8, y=-136.52), gain=1.5)

        apply_hand_brake = HandBrakeVehicle(self.other_vehicles[0], True)
        release_hand_brake = HandBrakeVehicle(self.other_vehicles[0], False)

        right_turn_trigger = InTriggerRegion(
            self.other_vehicles[0],
            75, 85,
            -150, -145)

        turn_right = BasicAgentBehavior(
            self.other_vehicles[0],
            carla.Location(x=65, y=-136.7, z=8))

        end_condition = InTriggerRegion(
            self.ego_vehicle,
            65, 68,
            -145, -125)

        root_timeout = TimeOut(self.timeout)

        # Creating non-leaf nodes
        root = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        scenario_sequence = py_trees.composites.Sequence()
        sync_arrival_parallel = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        # Building tree
        root.add_child(scenario_sequence)
        root.add_child(root_timeout)
        scenario_sequence.add_child(apply_hand_brake)
        scenario_sequence.add_child(start_other_trigger)
        scenario_sequence.add_child(release_hand_brake)
        scenario_sequence.add_child(sync_arrival_parallel)
        scenario_sequence.add_child(turn_right)
        scenario_sequence.add_child(end_condition)
        sync_arrival_parallel.add_child(sync_arrival)
        sync_arrival_parallel.add_child(right_turn_trigger)

        return root

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []

        # Adding checks for ego vehicle
        collision_criterion_ego = CollisionTest(self.ego_vehicle)
        region_check_ego = ReachedRegionTest(
            self.ego_vehicle,
            68, 72, -145, -135)
        criteria.append(collision_criterion_ego)
        criteria.append(region_check_ego)

        # Add approriate checks for other vehicles
        for vehicle in self.other_vehicles:
            collision_criterion = CollisionTest(vehicle)
            criteria.append(collision_criterion)

        return criteria
