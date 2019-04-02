#!/usr/bin/env python

#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Non-signalized junctions: crossing negotiation

The hero vehicle is passing through a junction without traffic lights
and encounters another vehicle passing across the junction.
"""

from queue import Queue

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

        self._world = world
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
        self._reference_waypoint = self._wmap.get_waypoint(config.ego_vehicle.transform.location)

        super(NoSignalJunctionCrossing, self).__init__("NoSignalJunctionCrossing",
                                                       ego_vehicle,
                                                       config,
                                                       world,
                                                       debug_mode,
                                                       criteria_enable=False)

    # def _initialize_actors(self, config):
    #     """
    #     Custom initialization
    #     """
    #     self._other_actor_transform = config.other_actors[0].transform
    #     first_vehicle_transform = carla.Transform(
    #         carla.Location(config.other_actors[0].transform.location.x,
    #                        config.other_actors[0].transform.location.y,
    #                        config.other_actors[0].transform.location.z - 5),
    #         config.other_actors[0].transform.rotation)
    #     first_vehicle = CarlaActorPool.request_new_actor(config.other_actors[0].model, first_vehicle_transform)
    #     self.other_actors.append(first_vehicle)

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

        left_boundary = generate_target_waypoint(self._reference_waypoint, turn=-1).transform.location
        right_boundary = generate_target_waypoint(self._reference_waypoint, turn=1).transform.location
        intersection_center = (left_boundary+right_boundary)/2
        intersection_width = left_boundary.distance(right_boundary)

        negotiation_queue_name = "ts10/detect_for_negotiation"
        passthrough_queue_name = "ts10/passthrough_queue"

        for queue_name in [negotiation_queue_name, passthrough_queue_name]:
            Blackboard().set(queue_name, Queue())

        # leaves
        root = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        scenario_sequence = py_trees.composites.Sequence()
        move_all_to_intersection = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)

        for i, actor in enumerate(self.other_actors):
            move_vehicle_to_intersection = py_trees.composites.Sequence()
            waypoint_follow_reach = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
            target_location = get_crossing_point(actor)
            waypoint_follow_reach.add_child(BasicAgentBehavior(actor, target_location))
            waypoint_follow_reach.add_child(InTriggerDistanceToNextIntersection(actor, 10))
            move_vehicle_to_intersection.add_child(waypoint_follow_reach)
            move_vehicle_to_intersection.add_child(StopVehicle(actor, 1.0))
            move_all_to_intersection.add_child(move_vehicle_to_intersection)

        scenario_sequence.add_child(move_all_to_intersection)

        detect_to_passthrough = DetectActorArrival(
            self._world, intersection_center, intersection_width, negotiation_queue_name)
        negotiate = PriorityNegotiator(negotiation_queue_name, passthrough_queue_name, interval=10)
        passthrough_follower = WaypointFollower(None, 30, blackboard_queue_name=passthrough_queue_name)

        root.add_child(scenario_sequence)
        root.add_child(detect_to_passthrough)
        root.add_child(negotiate)
        root.add_child(passthrough_follower)
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
