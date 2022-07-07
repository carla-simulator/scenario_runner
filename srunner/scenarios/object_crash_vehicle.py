#!/usr/bin/env python
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Object crash without prior vehicle action scenario:
The scenario realizes the user controlled ego vehicle
moving along the road and encountering a cyclist ahead.
"""

from __future__ import print_function

import math
import py_trees
import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorDestroy,
                                                                      KeepVelocity,
                                                                      Idle)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (InTriggerDistanceToLocation,
                                                                               InTimeToArrivalToLocation,
                                                                               DriveDistance)
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.scenario_helper import get_location_in_distance_from_wp

from srunner.tools.background_manager import LeaveSpaceInFront


class StationaryObjectCrossing(BasicScenario):

    """
    This class holds everything required for a simple object crash
    without prior vehicle action involving a vehicle and a cyclist.
    The ego vehicle is passing through a road and encounters
    a stationary cyclist.

    This is a single ego vehicle scenario
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=60):
        """
        Setup all relevant parameters and create scenario
        """
        self._wmap = CarlaDataProvider.get_map()
        self._reference_waypoint = self._wmap.get_waypoint(config.trigger_points[0].location)
        # ego vehicle parameters
        self._ego_vehicle_distance_driven = 40

        # other vehicle parameters
        self._other_actor_target_velocity = 10
        # Timeout of scenario in seconds
        self.timeout = timeout

        super(StationaryObjectCrossing, self).__init__("Stationaryobjectcrossing",
                                                       ego_vehicles,
                                                       config,
                                                       world,
                                                       debug_mode,
                                                       criteria_enable=criteria_enable)

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        _distance = 40
        lane_width = self._reference_waypoint.lane_width
        location, _ = get_location_in_distance_from_wp(self._reference_waypoint, _distance)
        waypoint = self._wmap.get_waypoint(location)
        offset = {"orientation": 270, "position": 90, "z": 0.4, "k": 0.2}
        position_yaw = waypoint.transform.rotation.yaw + offset['position']
        orientation_yaw = waypoint.transform.rotation.yaw + offset['orientation']
        offset_location = carla.Location(
            offset['k'] * lane_width * math.cos(math.radians(position_yaw)),
            offset['k'] * lane_width * math.sin(math.radians(position_yaw)))
        location += offset_location
        location.z += offset['z']
        self.transform = carla.Transform(location, carla.Rotation(yaw=orientation_yaw))
        static = CarlaDataProvider.request_new_actor('static.prop.container', self.transform)
        static.set_simulate_physics(True)
        self.other_actors.append(static)

    def _create_behavior(self):
        """
        Only behavior here is to wait
        """
        lane_width = self.ego_vehicles[0].get_world().get_map().get_waypoint(
            self.ego_vehicles[0].get_location()).lane_width
        lane_width = lane_width + (1.25 * lane_width)

        # leaf nodes
        actor_stand = Idle(15)
        actor_removed = ActorDestroy(self.other_actors[0])
        end_condition = DriveDistance(self.ego_vehicles[0], self._ego_vehicle_distance_driven)

        # non leaf nodes
        root = py_trees.composites.Parallel(
            name="StaticObstacle",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        scenario_sequence = py_trees.composites.Sequence()

        # building tree
        root.add_child(scenario_sequence)
        scenario_sequence.add_child(actor_stand)
        scenario_sequence.add_child(actor_removed)
        scenario_sequence.add_child(end_condition)

        return root

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []

        collision_criterion = CollisionTest(self.ego_vehicles[0])
        criteria.append(collision_criterion)

        return criteria

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()


class DynamicObjectCrossing(BasicScenario):

    """
    This class holds everything required for a simple object crash
    without prior vehicle action involving a vehicle and a cyclist/pedestrian,
    The ego vehicle is passing through a road,
    And encounters a cyclist/pedestrian crossing the road.

    This is a single ego vehicle scenario
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True, timeout=60):
        """
        Setup all relevant parameters and create scenario
        """
        self._wmap = CarlaDataProvider.get_map()
        self._trigger_location = config.trigger_points[0].location
        self._reference_waypoint = self._wmap.get_waypoint(self._trigger_location)
        self._num_lane_changes = 0

        if 'distance' in config.other_parameters:
            self._distance = int(config.other_parameters['distance']['value'])
        else:
            self._distance = 12

        if 'blocker_model' in config.other_parameters:
            self._blocker_model = config.other_parameters['blocker_model']['value']
        else:
            self._blocker_model = 'static.prop.vendingmachine'  # blueprint filter of the blocker

        if 'crossing_angle' in config.other_parameters:
            self._crossing_angle = float(config.other_parameters['crossing_angle']['value'])
        else:
            self._crossing_angle = 0  # Crossing angle of the pedestrian

        if abs(self._crossing_angle) > 90:
            raise ValueError("'crossing_angle' must be between -90 and 90º for the pedestrian to cross the road")

        self._blocker_shift = 0.9
        self._retry_dist = 0.4

        self._adversary_transform = None
        self._blocker_transform = None
        self._collision_wp = None

        self._adversary_speed = 2.0  # Speed of the adversary [m/s]
        self._reaction_time = 1.8  # Time the agent has to react to avoid the collision [s]
        self._min_trigger_dist = 6.0  # Min distance to the collision location that triggers the adversary [m]
        self._ego_end_distance = 40
        self.timeout = timeout

        self._number_of_attempts = 6

        super(DynamicObjectCrossing, self).__init__("DynamicObjectCrossing",
                                                    ego_vehicles,
                                                    config,
                                                    world,
                                                    debug_mode,
                                                    criteria_enable=criteria_enable)

    def _get_sidewalk_transform(self, waypoint, offset):
        """
        Processes the waypoint transform to find a suitable spawning one at the sidewalk.
        It first rotates the transform so that it is pointing towards the road and then moves a
        bit to the side waypoint that aren't part of sidewalks, as they might be invading the road
        """

        new_rotation = waypoint.transform.rotation
        new_rotation.yaw += offset['yaw']

        if waypoint.lane_type == carla.LaneType.Sidewalk:
            new_location = waypoint.transform.location
        else:
            right_vector = waypoint.transform.get_right_vector()
            offset_dist = offset["k"]
            offset_location = carla.Location(offset_dist * right_vector.x, offset_dist * right_vector.y)
            new_location = waypoint.transform.location + offset_location
        new_location.z += offset['z']

        return carla.Transform(new_location, new_rotation)

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        # Get the waypoint in front of the ego.
        move_dist = self._distance
        waypoint = self._reference_waypoint
        while self._number_of_attempts > 0:
            self._collision_dist = 0

            # Move to the front
            location, _ = get_location_in_distance_from_wp(waypoint, move_dist, False)
            waypoint = self._wmap.get_waypoint(location)
            self._collision_wp = waypoint

            # Move to the right
            sidewalk_waypoint = waypoint
            while sidewalk_waypoint.lane_type != carla.LaneType.Sidewalk:
                right_wp = sidewalk_waypoint.get_right_lane()
                if right_wp is None:
                    break  # No more right lanes
                sidewalk_waypoint = right_wp

            # Get the blocker transform and spawn it
            offset = {"yaw": 90, "z": 0.0, "k": 1.5}
            self._blocker_transform = self._get_sidewalk_transform(sidewalk_waypoint, offset)
            blocker = CarlaDataProvider.request_new_actor(self._blocker_model, self._blocker_transform)
            if not blocker:
                self._number_of_attempts -= 1
                move_dist = self._retry_dist
                print("Failed blocker")
                continue

            # Get the adversary transform and spawn it
            walker_dist = blocker.bounding_box.extent.x + 0.5
            wps = sidewalk_waypoint.next(walker_dist)
            if not wps:
                raise ValueError("Couldn't find a location to spawn the adversary")
            walker_wp = wps[0]

            offset = {"yaw": 270 - self._crossing_angle, "z": 0.5, "k": 1.2}
            self._adversary_transform = self._get_sidewalk_transform(walker_wp, offset)
            adversary = CarlaDataProvider.request_new_actor('walker.*', self._adversary_transform)
            if adversary is None:
                blocker.destroy()
                self._number_of_attempts -= 1
                move_dist = self._retry_dist
                print("Failed adversary")
                continue

            self._collision_dist += waypoint.transform.location.distance(self._adversary_transform.location)

            # Both actors were succesfully spawned, end
            break

        if self._number_of_attempts == 0:
            raise Exception("Couldn't find viable position for the adversary and blocker actors")

        blocker.set_simulate_physics(enabled=False)
        self.other_actors.append(adversary)
        self.other_actors.append(blocker)

    def _create_behavior(self):
        """
        After invoking this scenario, cyclist will wait for the user
        controlled vehicle to enter trigger distance region,
        the cyclist starts crossing the road once the condition meets,
        then after 60 seconds, a timeout stops the scenario
        """
        sequence = py_trees.composites.Sequence(name="CrossingActor")
        if self.route_mode:
            sequence.add_child(LeaveSpaceInFront(self._distance))

        collision_location = self._collision_wp.transform.location

        # Wait until ego is close to the adversary
        trigger_adversary = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, name="TriggerAdversaryStart")
        trigger_adversary.add_child(InTimeToArrivalToLocation(
            self.ego_vehicles[0], self._reaction_time, collision_location))
        trigger_adversary.add_child(InTriggerDistanceToLocation(
            self.ego_vehicles[0], collision_location, self._min_trigger_dist))
        sequence.add_child(trigger_adversary)

        # Move the adversary
        move_distance = 2 * self._collision_dist  # Cross the whole road (supposing symetry in both directions)
        move_duration = move_distance / self._adversary_speed
        sequence.add_child(KeepVelocity(
            self.other_actors[0], self._adversary_speed,
            duration=move_duration, distance=move_distance, name="AdversaryCrossing"))

        # Remove everything
        sequence.add_child(ActorDestroy(self.other_actors[0], name="DestroyAdversary"))
        sequence.add_child(ActorDestroy(self.other_actors[1], name="DestroyBlocker"))
        sequence.add_child(DriveDistance(self.ego_vehicles[0], self._ego_end_distance, name="EndCondition"))

        return sequence

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
        Remove all actors upon deletion
        """
        self.remove_all_actors()


class ParkingCrossingPedestrian(BasicScenario):

    """
    Variation of DynamicObjectCrossing but now the blocker is now a vehicle
    This class holds everything required for a simple object crash
    without prior vehicle action involving a vehicle and a cyclist/pedestrian,
    The ego vehicle is passing through a road,
    And encounters a cyclist/pedestrian crossing the road.

    This is a single ego vehicle scenario
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True, timeout=60):
        """
        Setup all relevant parameters and create scenario
        """
        self._wmap = CarlaDataProvider.get_map()
        self._trigger_location = config.trigger_points[0].location
        self._reference_waypoint = self._wmap.get_waypoint(self._trigger_location)
        self._num_lane_changes = 0

        if 'distance' in config.other_parameters:
            self._distance = int(config.other_parameters['distance']['value'])
        else:
            self._distance = 12

        if 'direction' in config.other_parameters:
            self._direction = config.other_parameters['direction']['value']
        else:
            self._direction = 'right'
        
        if self._direction not in ('right', 'left'):
            raise ValueError("'direction' value must be either 'left' or 'right'")

        self._adversary_speed = 3.0  # Speed of the adversary [m/s]
        self._reaction_time = 1.9  # Time the agent has to react to avoid the collision [s]
        self._min_trigger_dist = 6.0  # Min distance to the collision location that triggers the adversary [m]
        self._ego_end_distance = 40
        self.timeout = timeout

        super().__init__("ParkingCrossingPedestrian",
                         ego_vehicles,
                         config,
                         world,
                         debug_mode,
                         criteria_enable=criteria_enable)

    def _get_blocker_transform(self, waypoint):
        """Processes the driving wp to get a waypoint at the side that looks at the road"""
        if waypoint.lane_type == carla.LaneType.Sidewalk:
            new_location = waypoint.transform.location
        else:
            vector = waypoint.transform.get_right_vector()
            if self._direction == 'left':
                vector *= -1

            offset_location = carla.Location(waypoint.lane_width * vector.x, waypoint.lane_width * vector.y)
            new_location = waypoint.transform.location + offset_location
        new_location.z += 0.5

        return carla.Transform(new_location, waypoint.transform.rotation)

    def _get_walker_transform(self, waypoint):
        """Processes the driving wp to get a waypoint at the side that looks at the road"""

        new_rotation = waypoint.transform.rotation
        new_rotation.yaw += 270 if self._direction == 'right' else 90

        if waypoint.lane_type == carla.LaneType.Sidewalk:
            new_location = waypoint.transform.location
        else:
            vector = waypoint.transform.get_right_vector()
            if self._direction == 'left':
                vector *= -1

            offset_location = carla.Location(waypoint.lane_width * vector.x, waypoint.lane_width * vector.y)
            new_location = waypoint.transform.location + offset_location
        new_location.z += 0.5

        return carla.Transform(new_location, new_rotation)

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        # Get the adversary transform and spawn it
        wps = self._reference_waypoint.next(self._distance)
        if not wps:
            raise ValueError("Couldn't find a location to spawn the adversary")
        blocker_wp = wps[0]

        # Get the adversary transform and spawn it
        self._blocker_transform = self._get_blocker_transform(blocker_wp)
        blocker = CarlaDataProvider.request_new_actor('vehicle.*', self._blocker_transform, attribute_filter={'base_type': 'car'})
        if blocker is None:
            raise ValueError("Couldn't spawn the adversary")
        self.other_actors.append(blocker)
        blocker.apply_control(carla.VehicleControl(hand_brake=True))

        walker_dist = blocker.bounding_box.extent.x + 0.5
        wps = blocker_wp.next(walker_dist)
        if not wps:
            raise ValueError("Couldn't find a location to spawn the adversary")
        walker_wp = wps[0]

        # Get the adversary transform and spawn it
        self._walker_transform = self._get_walker_transform(walker_wp)
        walker = CarlaDataProvider.request_new_actor('walker.*', self._walker_transform)
        if walker is None:
            raise ValueError("Couldn't spawn the adversary")
        self.other_actors.append(walker)
        
        self._collision_wp = walker_wp

    def _create_behavior(self):
        """
        After invoking this scenario, cyclist will wait for the user
        controlled vehicle to enter trigger distance region,
        the cyclist starts crossing the road once the condition meets,
        then after 60 seconds, a timeout stops the scenario
        """
        sequence = py_trees.composites.Sequence(name="ParkingCrossingPedestrian")
        if self.route_mode:
            sequence.add_child(LeaveSpaceInFront(self._distance))

        collision_location = self._collision_wp.transform.location

        # Wait until ego is close to the adversary
        trigger_adversary = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, name="TriggerAdversaryStart")
        trigger_adversary.add_child(InTimeToArrivalToLocation(
            self.ego_vehicles[0], self._reaction_time, collision_location))
        trigger_adversary.add_child(InTriggerDistanceToLocation(
            self.ego_vehicles[0], collision_location, self._min_trigger_dist))
        sequence.add_child(trigger_adversary)

        # Move the adversary
        distance = 8.0  # Scenario is meant to be used at a one lane - one direction road
        duration = distance / self._adversary_speed

        sequence.add_child(KeepVelocity(
            self.other_actors[1], self._adversary_speed,
            duration=duration, distance=distance, name="AdversaryCrossing"))

        # Remove everything
        sequence.add_child(ActorDestroy(self.other_actors[0], name="DestroyAdversary"))
        sequence.add_child(ActorDestroy(self.other_actors[1], name="DestroyBlocker"))
        sequence.add_child(DriveDistance(self.ego_vehicles[0], self._ego_end_distance, name="EndCondition"))

        return sequence

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
        Remove all actors upon deletion
        """
        self.remove_all_actors()
