#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Non-signalized junctions: crossing negotiation

The hero vehicle is passing through a junction without traffic lights
and encounters another vehicle passing across the junction.
"""

import py_trees
from agents.navigation.local_planner import RoadOption

from srunner.scenariomanager.atomic_scenario_behavior import *
from srunner.scenariomanager.atomic_scenario_criteria import *
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenarios.basic_scenario import *
from srunner.scenarios.scenario_helper import *
from srunner.scenariomanager.timer import TimeOut


NO_SIGNAL_JUNCTION_SCENARIOS = [
    "NoSignalJunctionCrossing",
]


class NoSignalJunctionCrossing(BasicScenario):

    """
    Implementation class for
    'Non-signalized junctions: crossing negotiation' scenario,
    (Traffic Scenario 10).
    """

    category = "NoSignalJunction"
    timeout = 120

    def __init__(self, world, ego_vehicle, config, randomize=False, debug_mode=False, criteria_enable=True):
        """
        Setup all relevant parameters and create scenario
        """

        self._wmap = CarlaDataProvider.get_map()
        self._ego_vehicle_driven_distance = 100
        self._ego_vehicle_end_distance = 200
        self._other_actor_target_velocity = 50
        self._max_brake = 1.0
        self._arrival_threshold = 10
        self._ego_arrival_trigger = 15
        self._other_actor_wait_time = 10
        self._adversary_speed = 30
        self._end_threshold = 20

        super(NoSignalJunctionCrossing, self).__init__("NoSignalJunctionCrossing",
                                                       ego_vehicle,
                                                       config,
                                                       world,
                                                       debug_mode,
                                                       criteria_enable=False)

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        self._other_actor_transform = config.other_actors[0].transform
        first_vehicle_transform = carla.Transform(
            carla.Location(config.other_actors[0].transform.location.x,
                           config.other_actors[0].transform.location.y,
                           config.other_actors[0].transform.location.z - 5),
            config.other_actors[0].transform.rotation)
        first_vehicle = CarlaActorPool.request_new_actor(config.other_actors[0].model, first_vehicle_transform)
        self.other_actors.append(first_vehicle)

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

        root = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        scenario_sequence = py_trees.composites.Sequence()

        move_all_to_intersection = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)
        pass_through_all = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)

        for i, adversary in enumerate(self.other_actors):
            move_vehicle_to_intersection = py_trees.composites.Sequence()
            waypoint_follow_reach = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

            current_waypoint = self._wmap.get_waypoint(adversary.get_location())
            plan = self._make_plan(current_waypoint)

            waypoint_follow_reach.add_child(WaypointFollower(adversary, self._adversary_speed, plan))
            waypoint_follow_reach.add_child(InTriggerDistanceToNextIntersection(adversary, self._arrival_threshold))
            move_vehicle_to_intersection.add_child(waypoint_follow_reach)
            move_vehicle_to_intersection.add_child(StopVehicle(adversary, self._max_brake))
            move_all_to_intersection.add_child(move_vehicle_to_intersection)

            waypoint_beyond = generate_target_waypoint(current_waypoint, 0) # Straight across the intersection
            plan_beyond = self._make_plan(waypoint_beyond)
            wait_and_move = py_trees.composites.Sequence()
            wait_and_move.add_child(TimeOut(self._other_actor_wait_time*i))
            waypoint_follow_through = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
            waypoint_follow_through.add_child(WaypointFollower(adversary, self._adversary_speed, plan=plan_beyond))
            waypoint_follow_through.add_child(
                InTriggerDistanceToLocation(adversary, plan_beyond[-1][0].transform.location, self._end_threshold))
            wait_and_move.add_child(waypoint_follow_through)
            pass_through_all.add_child(wait_and_move)

        scenario_sequence.add_child(move_all_to_intersection)
        scenario_sequence.add_child(InTriggerDistanceToNextIntersection(self.ego_vehicle, self._ego_arrival_trigger))
        scenario_sequence.add_child(pass_through_all)

        root.add_child(scenario_sequence)
        root.add_child(DriveDistance(self.ego_vehicle, self._ego_vehicle_end_distance))

        return root

    def _make_plan(self, current_waypoint):
        """
        Creates a waypoint plan till the next intersection
        """
        plan = []
        wp_choice = current_waypoint.next(1.0)
        while not wp_choice[0].is_intersection:
            current_waypoint = wp_choice[0]
            plan.append((current_waypoint, RoadOption.LANEFOLLOW))
            wp_choice = current_waypoint.next(1.0)

        return plan

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []

        # Adding checks for ego vehicle
        collision_criterion_ego = CollisionTest(self.ego_vehicle)
        driven_distance_criterion = DrivenDistanceTest(self.ego_vehicle,
                                                       self._ego_vehicle_driven_distance,
                                                       distance_acceptable=90,
                                                       optional=True)
        criteria.append(collision_criterion_ego)
        criteria.append(driven_distance_criterion)

        return criteria

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()
