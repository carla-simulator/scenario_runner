#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Scenarios in which another (opposite) vehicle 'illegally' takes
priority, e.g. by running a red traffic light.
"""

from __future__ import print_function

import py_trees
import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorTransformSetter,
                                                                      ActorDestroy,
                                                                      TrafficLightFreezer,
                                                                      ConstantVelocityAgentBehavior)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (InTriggerDistanceToLocation,
                                                                               InTimeToArrivalToLocation,
                                                                               WaitEndIntersection)
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.scenario_helper import (get_geometric_linear_intersection,
                                           generate_target_waypoint,
                                           get_junction_topology,
                                           filter_junction_wp_direction)

from srunner.tools.background_manager import HandleJunctionScenario


class OppositeVehicleRunningRedLight(BasicScenario):
    """
    This class holds everything required for a scenario in which another vehicle runs a red light
    in front of the ego, forcing it to react. This vehicles are 'special' ones such as police cars,
    ambulances or firetrucks.
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=180):
        """
        Setup all relevant parameters and create scenario
        and instantiate scenario manager
        """
        self._world = world
        self._map = CarlaDataProvider.get_map()
        self._source_dist = 30
        self._sink_dist = 20

        if 'direction' in config.other_parameters:
            self._direction = config.other_parameters['direction']['value']
        else:
            self._direction = "right"

        if 'adversary_speed' in config.other_parameters:
            self._adversary_speed = float(config.other_parameters['flow_speed']['value'])
        else:
            self._adversary_speed = 70 / 3.6 # m/s

        self.timeout = timeout

        self._sync_time = 2.2  # Time the agent has to react to avoid the collision [s]
        self._min_trigger_dist = 9.0  # Min distance to the collision location that triggers the adversary [m]
        self._speed_duration_ratio = 2.0
        self._speed_distance_ratio = 1.5

        self._lights = carla.VehicleLightState.Special1 | carla.VehicleLightState.Special2

        super().__init__("OppositeVehicleRunningRedLight",
                         ego_vehicles,
                         config,
                         world,
                         debug_mode,
                         criteria_enable=criteria_enable)

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        ego_location = config.trigger_points[0].location
        self._ego_wp = CarlaDataProvider.get_map().get_waypoint(ego_location)

        # Get the junction
        starting_wp = self._ego_wp
        ego_junction_dist = 0
        while not starting_wp.is_junction:
            starting_wps = starting_wp.next(1.0)
            if len(starting_wps) == 0:
                raise ValueError("Failed to find junction as a waypoint with no next was detected")
            starting_wp = starting_wps[0]
            ego_junction_dist += 1
        junction = starting_wp.get_junction()

        # Get the opposite entry lane wp
        entry_wps, _ = get_junction_topology(junction)
        source_entry_wps = filter_junction_wp_direction(starting_wp, entry_wps, self._direction)
        if not source_entry_wps:
            raise ValueError("Couldn't find a lane for the given direction")

        # Get the source transform
        spawn_wp = source_entry_wps[0]
        source_junction_dist = 0
        while source_junction_dist < self._source_dist:
            spawn_wps = spawn_wp.previous(1.0)
            if len(spawn_wps) == 0:
                raise ValueError("Failed to find a source location as a waypoint with no previous was detected")
            if spawn_wps[0].is_junction:
                break
            spawn_wp = spawn_wps[0]
            source_junction_dist += 1
        self._spawn_wp = spawn_wp

        source_transform = spawn_wp.transform
        self._spawn_location = carla.Transform(
            source_transform.location + carla.Location(z=0.1),
            source_transform.rotation
        )

        # Spawn the actor and move it below ground
        opposite_actor = CarlaDataProvider.request_new_actor(
            'vehicle.*', self._spawn_location, attribute_filter={'special_type': 'emergency'})
        if not opposite_actor:
            raise Exception("Couldn't spawn the actor")
        lights = opposite_actor.get_light_state()
        lights |= self._lights
        opposite_actor.set_light_state(carla.VehicleLightState(lights))
        self.other_actors.append(opposite_actor)

        opposite_transform = carla.Transform(
            source_transform.location - carla.Location(z=500),
            source_transform.rotation
        )
        opposite_actor.set_transform(opposite_transform)
        opposite_actor.set_simulate_physics(enabled=False)

        # Get the sink location
        sink_exit_wp = generate_target_waypoint(self._map.get_waypoint(source_transform.location), 0)
        sink_wps = sink_exit_wp.next(self._sink_dist)
        if len(sink_wps) == 0:
            raise ValueError("Failed to find a sink location as a waypoint with no next was detected")
        self._sink_wp = sink_wps[0]

        # get the collision location
        self._collision_location = get_geometric_linear_intersection(
            starting_wp.transform.location, source_entry_wps[0].transform.location, True)
        if not self._collision_location:
            raise ValueError("Couldn't find an intersection point")

        # Get the z component
        collision_wp = self._map.get_waypoint(self._collision_location)
        self._collision_location.z = collision_wp.transform.location.z

        self._get_traffic_lights(junction, ego_junction_dist, source_junction_dist)

    def _get_traffic_lights(self, junction, ego_dist, source_dist):
        """Get the traffic light of the junction, mapping their states"""
        tls = self._world.get_traffic_lights_in_junction(junction.id)
        if not tls:
            raise ValueError("No traffic lights found, use the NonSignalized version instead")

        ego_landmark = self._ego_wp.get_landmarks_of_type(ego_dist + 2, "1000001")[0]
        ego_tl = self._world.get_traffic_light(ego_landmark)
        source_landmark = self._spawn_wp.get_landmarks_of_type(source_dist + 2, "1000001")[0]
        source_tl = self._world.get_traffic_light(source_landmark)
        self._tl_dict = {}
        for tl in tls:
            if tl.id in (ego_tl.id, source_tl.id):
                self._tl_dict[tl] = carla.TrafficLightState.Green
            else:
                self._tl_dict[tl] = carla.TrafficLightState.Red

    def _create_behavior(self):
        """
        Hero vehicle is entering a junction in an urban area, at a signalized intersection,
        while another actor runs a red lift, forcing the ego to break.
        """
        sequence = py_trees.composites.Sequence(name="OppositeVehicleTakingPriority")

        # Wait until ego is close to the adversary
        trigger_adversary = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, name="TriggerAdversaryStart")
        trigger_adversary.add_child(InTimeToArrivalToLocation(
            self.ego_vehicles[0], self._sync_time, self._collision_location))
        trigger_adversary.add_child(InTriggerDistanceToLocation(
            self.ego_vehicles[0], self._collision_location, self._min_trigger_dist))

        sequence.add_child(trigger_adversary)
        sequence.add_child(ConstantVelocityAgentBehavior(
            self.other_actors[0], target_location=self._sink_wp.transform.location,
            target_speed=self._adversary_speed, opt_dict={'ignore_vehicles': True, 'ignore_traffic_lights': True}, name="AdversaryCrossing"))

        main_behavior = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        main_behavior.add_child(TrafficLightFreezer(self._tl_dict))
        main_behavior.add_child(sequence)

        root = py_trees.composites.Sequence()
        if self.route_mode:
            root.add_child(HandleJunctionScenario(
                clear_junction=True,
                clear_ego_entry=True,
                remove_entries=[self._spawn_wp],
                remove_exits=[self._sink_wp],
                stop_entries=False,
                extend_road_exit=0
            ))
        root.add_child(ActorTransformSetter(self.other_actors[0], self._spawn_location))
        root.add_child(main_behavior)
        root.add_child(ActorDestroy(self.other_actors[0]))
        root.add_child(WaitEndIntersection(self.ego_vehicles[0]))

        return root

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        if self.route_mode:
            return []
        return [CollisionTest(self.ego_vehicles[0])]

    def __del__(self):
        """
        Remove all actors and traffic lights upon deletion
        """
        self.remove_all_actors()


class OppositeVehicleTakingPriority(BasicScenario):
    """
    This class holds everything required for a scenario in which another vehicle takes
    priority at an intersection, forcing the ego to break.
    This vehicles are 'special' ones such as police cars, ambulances or firetrucks.
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=180):
        """
        Setup all relevant parameters and create scenario
        and instantiate scenario manager
        """
        self._world = world
        self._map = CarlaDataProvider.get_map()
        self._source_dist = 30
        self._sink_dist = 20

        if 'direction' in config.other_parameters:
            self._direction = config.other_parameters['direction']['value']
        else:
            self._direction = "right"

        if 'adversary_speed' in config.other_parameters:
            self._adversary_speed = float(config.other_parameters['flow_speed']['value'])
        else:
            self._adversary_speed = 70 / 3.6 # m/s

        self._opposite_bp_wildcards = ['*firetruck*', '*ambulance*', '*police*']  # Wildcard patterns of the blueprints
        self.timeout = timeout

        self._sync_time = 2.2  # Time the agent has to react to avoid the collision [s]
        self._min_trigger_dist = 9.0  # Min distance to the collision location that triggers the adversary [m]
        self._speed_duration_ratio = 2.0
        self._speed_distance_ratio = 1.5

        self._lights = carla.VehicleLightState.Special1 | carla.VehicleLightState.Special2

        # Get the CDP seed or at routes, all copies of the scenario will have the same configuration
        self._rng = CarlaDataProvider.get_random_seed()

        super().__init__("OppositeVehicleTakingPriority",
                         ego_vehicles,
                         config,
                         world,
                         debug_mode,
                         criteria_enable=criteria_enable)

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        ego_location = config.trigger_points[0].location
        self._ego_wp = CarlaDataProvider.get_map().get_waypoint(ego_location)

        # Get the junction
        starting_wp = self._ego_wp
        ego_junction_dist = 0
        while not starting_wp.is_junction:
            starting_wps = starting_wp.next(1.0)
            if len(starting_wps) == 0:
                raise ValueError("Failed to find junction as a waypoint with no next was detected")
            starting_wp = starting_wps[0]
            ego_junction_dist += 1
        junction = starting_wp.get_junction()

        # Get the opposite entry lane wp
        entry_wps, _ = get_junction_topology(junction)
        source_entry_wps = filter_junction_wp_direction(starting_wp, entry_wps, self._direction)
        if not source_entry_wps:
            raise ValueError("Couldn't find a lane for the given direction")

        # Get the source transform
        spawn_wp = source_entry_wps[0]
        source_junction_dist = 0
        while source_junction_dist < self._source_dist:
            spawn_wps = spawn_wp.previous(1.0)
            if len(spawn_wps) == 0:
                raise ValueError("Failed to find a source location as a waypoint with no previous was detected")
            if spawn_wps[0].is_junction:
                break
            spawn_wp = spawn_wps[0]
            source_junction_dist += 1
        self._spawn_wp = spawn_wp

        source_transform = spawn_wp.transform
        self._spawn_location = carla.Transform(
            source_transform.location + carla.Location(z=0.1),
            source_transform.rotation
        )

        # Spawn the actor and move it below ground
        opposite_bp_wildcard = self._rng.choice(self._opposite_bp_wildcards)
        opposite_actor = CarlaDataProvider.request_new_actor(opposite_bp_wildcard, self._spawn_location)
        if not opposite_actor:
            raise Exception("Couldn't spawn the actor")
        opposite_actor.set_light_state(carla.VehicleLightState(
            carla.VehicleLightState.Special1 | carla.VehicleLightState.Special2))
        self.other_actors.append(opposite_actor)

        opposite_transform = carla.Transform(
            source_transform.location - carla.Location(z=500),
            source_transform.rotation
        )
        opposite_actor.set_transform(opposite_transform)
        opposite_actor.set_simulate_physics(enabled=False)

        # Get the sink location
        sink_exit_wp = generate_target_waypoint(self._map.get_waypoint(source_transform.location), 0)
        sink_wps = sink_exit_wp.next(self._sink_dist)
        if len(sink_wps) == 0:
            raise ValueError("Failed to find a sink location as a waypoint with no next was detected")
        self._sink_wp = sink_wps[0]

        # get the collision location
        self._collision_location = get_geometric_linear_intersection(
            starting_wp.transform.location, source_entry_wps[0].transform.location, True)
        if not self._collision_location:
            raise ValueError("Couldn't find an intersection point")

        # Get the z component
        collision_wp = self._map.get_waypoint(self._collision_location)
        self._collision_location.z = collision_wp.transform.location.z

    def _create_behavior(self):
        """
        Hero vehicle is entering a junction in an urban area, at a signalized intersection,
        while another actor runs a red lift, forcing the ego to break.
        """
        sequence = py_trees.composites.Sequence(name="OppositeVehicleTakingPriority")

        # Wait until ego is close to the adversary
        trigger_adversary = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, name="TriggerAdversaryStart")
        trigger_adversary.add_child(InTimeToArrivalToLocation(
            self.ego_vehicles[0], self._sync_time, self._collision_location))
        trigger_adversary.add_child(InTriggerDistanceToLocation(
            self.ego_vehicles[0], self._collision_location, self._min_trigger_dist))

        sequence.add_child(trigger_adversary)
        sequence.add_child(ConstantVelocityAgentBehavior(
            self.other_actors[0], target_location=self._sink_wp.transform.location,
            target_speed=self._adversary_speed, opt_dict={'ignore_vehicles': True, 'ignore_traffic_lights': True}, name="AdversaryCrossing"))

        main_behavior = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        main_behavior.add_child(sequence)

        root = py_trees.composites.Sequence()
        if self.route_mode:
            root.add_child(HandleJunctionScenario(
                clear_junction=True,
                clear_ego_entry=True,
                remove_entries=[self._spawn_wp],
                remove_exits=[self._sink_wp],
                stop_entries=True,
                extend_road_exit=0
            ))

        root.add_child(ActorTransformSetter(self.other_actors[0], self._spawn_location))
        root.add_child(main_behavior)
        root.add_child(ActorDestroy(self.other_actors[0]))
        root.add_child(WaitEndIntersection(self.ego_vehicles[0]))

        return root

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        if self.route_mode:
            return []
        return [CollisionTest(self.ego_vehicles[0])]

    def __del__(self):
        """
        Remove all actors and traffic lights upon deletion
        """
        self.remove_all_actors()
