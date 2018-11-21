#!/usr/bin/env python

#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Welcome to CARLA following vehicle scenario.

The scenario realizes a common driving behavior, in which the
user-controlled ego vehicle follows a leading car driving down
a given road. At some point the leading car has to slow down and
finally stop. The ego vehicle has to react accordingly to avoid
a collision.

"""

import random

import py_trees
import carla

from ScenarioManager import atomic_scenario_behavior
from ScenarioManager import scenario_manager
from ScenarioManager import atomic_scenario_criteria


class FollowLeadingVehicle(object):

    """
    This class holds everything required for a simple "Follow a leading vehicle"
    scenario involving two vehicles.
    """

    manager = None          # Scenario manager
    criteria_list = []      # List of evaluation criteria
    timeout = 10            # Timeout of scenario in seconds

    # ego vehicle parameters
    ego_vehicle = None
    ego_vehicle_model = 'vehicle.carlamotors.carlacola'
    ego_vehicle_start = carla.Transform(
        carla.Location(x=312, y=129, z=39), carla.Rotation(yaw=180))
    ego_vehicle_max_velocity_allowed = 20   # Maximum allowed velocity

    # other vehicle
    other_vehicle = None
    other_vehicle_model = 'vehicle.tesla.model3'
    other_vehicle_start = carla.Transform(
        carla.Location(x=263, y=129, z=39), carla.Rotation(yaw=180))
    other_vehicle_target_velocity = 15      # Target velocity of other vehicle
    trigger_distance_from_ego_vehicle = 15  # Starting point of other vehicle maneuver
    other_vehicle_max_throttle = 1.0        # Maximum throttle of other vehicle
    other_vehicle_max_brake = 1.0           # Maximum brake of other vehicle

    def __init__(self, world):
        """
        Setup all relevant parameters and create scenario
        and instantiate scenario manager
        """
        self.other_vehicle = self.setup_vehicle(world,
                                                self.other_vehicle_model,
                                                self.other_vehicle_start)
        self.ego_vehicle = self.setup_vehicle(world,
                                              self.ego_vehicle_model,
                                              self.ego_vehicle_start)

        # Setup scenario
        # py_trees.logging.level = py_trees.logging.Level.DEBUG
        behavior = self.create_behavior()
        criteria = self.create_test_criteria()
        scenario = scenario_manager.Scenario(behavior, criteria, self.timeout)
        self.manager = scenario_manager.ScenarioManager(scenario)

    def execute(self):
        """
        Function to run and analyze the scenario
        """
        self.manager.run_scenario()
        if not self.manager.analyze_scenario():
            print("Success!")
        else:
            print("Failure!")

        print("\n")
        print("\n")

    def setup_vehicle(self, world, model, spawn_point):
        """
        Function to setup the most relevant vehicle parameters,
        incl. spawn point and vehicle model.
        """
        blueprint_library = world.get_blueprint_library()

        # Get vehicle by model
        bp = random.choice(blueprint_library.filter(model))
        color = random.choice(bp.get_attribute('color').recommended_values)
        bp.set_attribute('color', color)
        vehicle = world.spawn_actor(bp, spawn_point)

        # Let's put the vehicle to drive around.
        vehicle.set_autopilot(False)

        return vehicle

    def create_behavior(self):
        """
        Example of a user defined scenario behavior. This function should be
        adapted by the user for other scenarios.

        The scenario defined after is a "follow leading vehicle" scenario. After
        invoking this scenario, it will wait for the user controlled vehicle to
        enter the start region, then make a traffic participant to accelerate
        until reaching a certain speed, then keep this speed for 2 seconds,
        before initiating a stopping maneuver. Finally, the user-controlled
        vehicle has to reach a target region.
        If this does not happen within 60 seconds, a timeout stops the scenario
        """

        # Build behavior tree
        sequence = py_trees.composites.Sequence("Sequence Behavior")
        startcondition = atomic_scenario_behavior.InTriggerDistance(
            self.other_vehicle,
            self.ego_vehicle,
            self.trigger_distance_from_ego_vehicle,
            name="Waiting for start position")
        accelerate = atomic_scenario_behavior.AccelerateToVelocity(
            self.other_vehicle,
            self.other_vehicle_max_throttle,
            self.other_vehicle_target_velocity)
        stop = atomic_scenario_behavior.StopVehicle(
            self.other_vehicle,
            self.other_vehicle_max_brake)
        keep_velocity = atomic_scenario_behavior.KeepVelocity(
            self.other_vehicle,
            self.other_vehicle_target_velocity,
            duration=2)
        endcondition = atomic_scenario_behavior.InTriggerRegion(
            self.ego_vehicle, 198, 200, 128, 130, name="Waiting for end position")

        sequence.add_child(startcondition)
        sequence.add_child(accelerate)
        sequence.add_child(keep_velocity)
        sequence.add_child(stop)
        sequence.add_child(endcondition)

        return sequence

    def create_test_criteria(self):
        """
        Example of a user defined test catalogue.
        This function should be adapted by the user.

        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []

        max_velocity_criterion = atomic_scenario_criteria.MaxVelocityTest(
            self.ego_vehicle,
            self.ego_vehicle_max_velocity_allowed)
        collision_criterion = atomic_scenario_criteria.CollisionTest(
            self.ego_vehicle)
        keep_lane_criterion = atomic_scenario_criteria.KeepLaneTest(
            self.ego_vehicle)

        criteria.append(max_velocity_criterion)
        criteria.append(collision_criterion)
        criteria.append(keep_lane_criterion)

        return criteria

    def __del__(self):
        """
        Cleanup.
        - Scenario manager stops the scenario and triggers a cleanup.
        - Removal of the vehicles
        """
        self.manager.stop_scenario()

        actors = [self.ego_vehicle, self.other_vehicle]
        for actor in actors:
            actor.destroy()
            actor = None


def main():
    """
    Main function starting a CARLA client and connecting to the world.
    """
    world = None

    try:
        # First of all, we need to create the client that will send the requests
        # to the simulator. Here we'll assume the simulator is accepting
        # requests in the localhost at port 2000.
        client = carla.Client('localhost', 2000)
        client.set_timeout(2.0)

        # Once we have a client we can retrieve the world that is currently
        # running.
        world = client.get_world()

        # Wait for the world to be ready
        world.wait_for_tick(10.0)

        follow_leading_vehicle = FollowLeadingVehicle(world)
        follow_leading_vehicle.execute()

    finally:
        if world is not None:
            del world


if __name__ == '__main__':

    main()
