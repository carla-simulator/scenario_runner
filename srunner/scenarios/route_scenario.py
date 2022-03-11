#!/usr/bin/env python

# Copyright (c) 2019-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides Challenge routes as standalone scenarios
"""

from __future__ import print_function

import traceback

import py_trees

from numpy import random
import carla

from agents.navigation.local_planner import RoadOption

from srunner.scenarioconfigs.scenario_configuration import ActorConfigurationData
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import ScenarioTriggerer
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import WaitForBlackboardVariable
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.route_parser import RouteParser, TRIGGER_THRESHOLD
from srunner.tools.route_manipulation import interpolate_trajectory
from srunner.tools.py_trees_port import oneshot_behavior

from srunner.scenarios.control_loss import ControlLoss
from srunner.scenarios.follow_leading_vehicle import FollowLeadingVehicleRoute
from srunner.scenarios.object_crash_vehicle import DynamicObjectCrossing
from srunner.scenarios.object_crash_intersection import VehicleTurningRoute
from srunner.scenarios.other_leading_vehicle import OtherLeadingVehicle
from srunner.scenarios.maneuver_opposite_direction import ManeuverOppositeDirection
from srunner.scenarios.junction_crossing_route import NoSignalJunctionCrossingRoute
from srunner.scenarios.signalized_junction_left_turn import SignalizedJunctionLeftTurn
from srunner.scenarios.signalized_junction_right_turn import SignalizedJunctionRightTurn
from srunner.scenarios.opposite_vehicle_taking_priority import OppositeVehicleRunningRedLight
from srunner.scenarios.background_activity import BackgroundActivity

from srunner.scenariomanager.scenarioatomics.atomic_criteria import (CollisionTest,
                                                                     InRouteTest,
                                                                     RouteCompletionTest,
                                                                     OutsideRouteLanesTest,
                                                                     RunningRedLightTest,
                                                                     RunningStopTest,
                                                                     ActorBlockedTest)

SECONDS_GIVEN_PER_METERS = 0.4

NUMBER_CLASS_TRANSLATION = {
    "Scenario1": ControlLoss,
    "Scenario2": FollowLeadingVehicleRoute,
    "Scenario3": DynamicObjectCrossing,
    "Scenario4": VehicleTurningRoute,
    "Scenario5": OtherLeadingVehicle,
    "Scenario6": ManeuverOppositeDirection,
    "Scenario7": OppositeVehicleRunningRedLight,
    "Scenario8": SignalizedJunctionLeftTurn,
    "Scenario9": SignalizedJunctionRightTurn,
    "Scenario10": NoSignalJunctionCrossingRoute
}


class RouteScenario(BasicScenario):

    """
    Implementation of a RouteScenario, i.e. a scenario that consists of driving along a pre-defined route,
    along which several smaller scenarios are triggered
    """

    def __init__(self, world, config, debug_mode=False, criteria_enable=True, timeout=300):
        """
        Setup all relevant parameters and create scenarios along route
        """

        self.config = config
        self.route = self._get_route(config)
        sampled_scenario_definitions = self._get_scenarios(config)
        ego_vehicle = self._spawn_ego_vehicle()
        self.timeout = self._estimate_route_timeout()

        if debug_mode:
            self._draw_waypoints(world, self.route, vertical_shift=0.1, size=0.1, persistency=self.timeout)

        self._build_scenarios(
            world, ego_vehicle, sampled_scenario_definitions, 5, self.timeout, debug_mode > 0
        )

        super(RouteScenario, self).__init__(
            config.name, [ego_vehicle], config, world, debug_mode > 1, False, criteria_enable
        )

    def _get_route(self, config):
        """
        Gets the route from the configuration, interpolating it to the desired density,
        saving it to the CarlaDataProvider and sending it to the agent

        Parameters:
        - world: CARLA world
        - config: Scenario configuration (RouteConfiguration)
        - debug_mode: boolean to decide whether or not the route poitns are printed
        """
        # prepare route's trajectory (interpolate and add the GPS route)
        gps_route, route = interpolate_trajectory(config.trajectory)
        if config.agent is not None:
            config.agent.set_global_plan(gps_route, route)

        return route

    def _get_scenarios(self, config):
        """
        Gets the scenarios that will be part of the route. Automatically filters the scenarios
        that affect the route and, if there are two or more scenarios with very similar triggering positions,
        one of those is randomly chosen

        Parameters:
        - config: Scenario configuration (RouteConfiguration)
        """
        scenario_dict = RouteParser.parse_scenario_file_to_dict(config.scenario_file)
        potential_scenarios = RouteParser.scan_route_for_scenarios(config.town, self.route, scenario_dict)
        sampled_scenarios = self._scenario_sampling(potential_scenarios)
        return sampled_scenarios

    def _spawn_ego_vehicle(self):
        """Spawn the ego vehicle at the first waypoint of the route"""
        elevate_transform = self.route[0][0]
        elevate_transform.location.z += 0.5

        ego_vehicle = CarlaDataProvider.request_new_actor('vehicle.lincoln.mkz_2017',
                                                          elevate_transform,
                                                          rolename='hero')

        return ego_vehicle

    def _estimate_route_timeout(self):
        """
        Estimate the duration of the route, as a proportinal value of its length
        """
        route_length = 0.0  # in meters

        prev_point = self.route[0][0]
        for current_point, _ in self.route[1:]:
            dist = current_point.location.distance(prev_point.location)
            route_length += dist
            prev_point = current_point

        return int(SECONDS_GIVEN_PER_METERS * route_length)

    # pylint: disable=no-self-use
    def _draw_waypoints(self, world, waypoints, vertical_shift, size, persistency=-1):
        """
        Draw a list of waypoints at a certain height given in vertical_shift.
        """
        for w in waypoints:
            wp = w[0].location + carla.Location(z=vertical_shift)

            if w[1] == RoadOption.LEFT:  # Yellow
                color = carla.Color(128, 128, 0)
            elif w[1] == RoadOption.RIGHT:  # Cyan
                color = carla.Color(0, 128, 128)
            elif w[1] == RoadOption.CHANGELANELEFT:  # Orange
                color = carla.Color(128, 32, 0)
            elif w[1] == RoadOption.CHANGELANERIGHT:  # Dark Cyan
                color = carla.Color(0, 32, 128)
            elif w[1] == RoadOption.STRAIGHT:  # Gray
                color = carla.Color(64, 64, 64)
            else:  # LANEFOLLOW
                color = carla.Color(0, 128, 0)  # Green

            world.debug.draw_point(wp, size=0.1, color=color, life_time=persistency)

        world.debug.draw_point(waypoints[0][0].location + carla.Location(z=vertical_shift), size=2*size,
                               color=carla.Color(0, 0, 128), life_time=persistency)
        world.debug.draw_point(waypoints[-1][0].location + carla.Location(z=vertical_shift), size=2*size,
                               color=carla.Color(128, 0, 0), life_time=persistency)

    def _scenario_sampling(self, potential_scenarios, random_seed=0):
        """Sample the scenarios that are going to happen for this route."""
        # Fix the random seed for reproducibility, and randomly sample a scenario per trigger position.
        rng = random.RandomState(random_seed)

        sampled_scenarios = []
        for trigger in list(potential_scenarios):
            scenario_list = potential_scenarios[trigger]
            sampled_scenarios.append(rng.choice(scenario_list))

        return sampled_scenarios

    def _build_scenarios(self, world, ego_vehicle, scenario_definitions,
                         scenarios_per_tick=5, timeout=300, debug_mode=False):
        """
        Initializes the class of all the scenarios that will be present in the route.
        If a class fails to be initialized, a warning is printed but the route execution isn't stopped
        """
        self.list_scenarios = []

        if debug_mode:
            tmap = CarlaDataProvider.get_map()
            for scenario_config in scenario_definitions:
                scenario_loc = scenario_config.trigger_points[0].location
                debug_loc = tmap.get_waypoint(scenario_loc).transform.location + carla.Location(z=0.2)
                world.debug.draw_point(debug_loc, size=0.2, color=carla.Color(128, 0, 0), life_time=timeout)
                world.debug.draw_string(debug_loc, str(scenario_config.type), draw_shadow=False,
                                        color=carla.Color(0, 0, 128), life_time=timeout, persistent_lines=True)

        for scenario_number, scenario_config in enumerate(scenario_definitions):
            scenario_config.ego_vehicles = [ActorConfigurationData(ego_vehicle.type_id,
                                                                   ego_vehicle.get_transform(),
                                                                   'hero')]
            scenario_config.route_var_name = "ScenarioRouteNumber{}".format(scenario_number)

            try:
                scenario_class = NUMBER_CLASS_TRANSLATION[scenario_config.type]
                scenario_instance = scenario_class(world, [ego_vehicle], scenario_config,
                                                   criteria_enable=False, timeout=timeout)

                # Do a tick every once in a while to avoid spawning everything at the same time
                if scenario_number % scenarios_per_tick == 0:
                    if CarlaDataProvider.is_sync_mode():
                        world.tick()
                    else:
                        world.wait_for_tick()

            except Exception as e:      # pylint: disable=broad-except
                if debug_mode:
                    traceback.print_exc()
                print("Skipping scenario '{}' due to setup error: {}".format(scenario_config.type, e))
                continue

            self.list_scenarios.append(scenario_instance)

    # pylint: enable=no-self-use
    def _initialize_actors(self, config):
        """
        Set other_actors to the superset of all scenario actors
        """
        # Add all the actors of the specific scenarios to self.other_actors
        for scenario in self.list_scenarios:
            self.other_actors.extend(scenario.other_actors)

    def _create_behavior(self):
        """
        Creates a parallel behavior that runs all of the scenarios part of the route.
        These subbehaviors have had a trigger condition added so that they wait until
        the agent is close to their trigger point before activating.

        It also adds the BackgroundActivity scenario, which will be active throughout the whole route.
        This behavior never ends and the end condition is given by the RouteCompletionTest criterion.
        """
        scenario_trigger_distance = TRIGGER_THRESHOLD  # Max trigger distance between route and scenario

        behavior = py_trees.composites.Parallel(name="Route Behavior",
                                                policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)

        scenario_behaviors = []
        blackboard_list = []

        for scenario in self.list_scenarios:
            if scenario.behavior_tree is not None:
                scenario_behaviors.append(scenario.behavior_tree)
                blackboard_list.append([scenario.config.route_var_name,
                                        scenario.config.trigger_points[0].location])

        # Add the behavior that manages the scenario trigger conditions
        scenario_triggerer = ScenarioTriggerer(
            self.ego_vehicles[0], self.route, blackboard_list, scenario_trigger_distance, repeat_scenarios=False
        )
        behavior.add_child(scenario_triggerer)  # Tick the ScenarioTriggerer before the scenarios

        # Add the Background Activity
        background_activity = BackgroundActivity(
            self.world, self.ego_vehicles[0], self.config, self.route, self.night_mode, timeout=self.timeout
        )
        behavior.add_child(background_activity.behavior_tree)

        behavior.add_children(scenario_behaviors)
        return behavior

    def _create_test_criteria(self):
        """
        Create the criteria tree. It starts with some route criteria (which are always active),
        and adds the scenario specific ones, which will only be active during their scenario
        """
        criteria = py_trees.composites.Parallel(name="Criteria",
                                                policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)

        # End condition
        criteria.add_child(RouteCompletionTest(self.ego_vehicles[0], route=self.route))

        # 'Normal' criteria
        criteria.add_child(OutsideRouteLanesTest(self.ego_vehicles[0], route=self.route))
        criteria.add_child(CollisionTest(self.ego_vehicles[0], other_actor_type='vehicle', name="CollisionVehicleTest"))
        criteria.add_child(CollisionTest(self.ego_vehicles[0], other_actor_type='miscellaneous', name="CollisionLayoutTest"))
        criteria.add_child(CollisionTest(self.ego_vehicles[0], other_actor_type='walker', name="CollisionPedestrianTest"))
        criteria.add_child(RunningRedLightTest(self.ego_vehicles[0]))
        criteria.add_child(RunningStopTest(self.ego_vehicles[0]))

        # These stop the route early to save computational time
        criteria.add_child(InRouteTest(
            self.ego_vehicles[0], route=self.route, offroad_max=30, terminate_on_failure=True))
        criteria.add_child(ActorBlockedTest(
            self.ego_vehicles[0], min_speed=0.1, max_time=180.0, terminate_on_failure=True, name="AgentBlockedTest")
        )

        for scenario in self.list_scenarios:

            scenario_criteria = scenario.get_criteria()
            if len(scenario_criteria) == 0:
                continue  # No need to create anything

            criteria_tree = self._create_criterion_tree(scenario,
                scenario_criteria,
            )
            criteria.add_child(criteria_tree)

        return criteria

    def _create_criterion_tree(self, scenario, criteria):
        """
        We can make use of the blackboard variables used by the behaviors themselves,
        as we already have an atomic that handles their (de)activation.
        The criteria will wait until that variable is active (the scenario has started),
        and will automatically stop when it deactivates (as the scenario has finished)
        """
        scenario_name = scenario.name,
        var_name = scenario.config.route_var_name
        check_name = "WaitForBlackboardVariable: {}".format(var_name)

        criteria_tree = py_trees.composites.Sequence(name=scenario_name)
        criteria_tree.add_child(WaitForBlackboardVariable(var_name, True, False, name=check_name))

        scenario_criteria = py_trees.composites.Parallel(name=scenario_name,
                                                policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        for criterion in criteria:
            scenario_criteria.add_child(criterion)
        scenario_criteria.add_child(WaitForBlackboardVariable(var_name, False, None, name=check_name))

        criteria_tree.add_child(scenario_criteria)
        return criteria_tree


    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()
