#!/usr/bin/env python
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Leaderboard-2.1-era definitions of the route scenarios whose current versions
require actor-flow parameters ('start_actor_flow'/'end_actor_flow') that no
published route file provides. RouteScenario falls back to these classes when
a route was authored against the old parameter set, instead of skipping the
scenario altogether.

The classes are verbatim ports of the origin/leaderboard-2.1 versions of
construction_crash_vehicle.py and parking_exit.py, renamed with a 'Legacy'
suffix so both generations can coexist in the scenario registry.
"""

from __future__ import print_function

import py_trees
import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorDestroy,
                                                                      ActorTransformSetter,
                                                                      SwitchWrongDirectionTest,
                                                                      ScenarioTimeout,
                                                                      Idle, WaitForever,
                                                                      ChangeAutoPilot,
                                                                      OppositeActorFlow)
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (InTriggerDistanceToLocation,
                                                                               WaitUntilInFrontPosition,
                                                                               DriveDistance)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest, ScenarioTimeoutTest
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.background_manager import (RemoveRoadLane,
                                              ReAddRoadLane,
                                              SetMaxSpeed,
                                              ChangeOppositeBehavior,
                                              ChangeRoadBehavior)


def get_value_parameter(config, name, p_type, default):
    if name in config.other_parameters:
        return p_type(config.other_parameters[name]['value'])
    else:
        return default


def get_interval_parameter(config, name, p_type, default):
    if name in config.other_parameters:
        return [
            p_type(config.other_parameters[name]['from']),
            p_type(config.other_parameters[name]['to'])
        ]
    else:
        return default


class ConstructionObstacleLegacy(BasicScenario):
    """
    This class holds everything required for a construction scenario
    The ego vehicle is passing through a road and encounters
    a stationary rectangular construction cones setup and traffic warning,
    forcing it to lane change.

    This is a single ego vehicle scenario
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False,
                 criteria_enable=True, timeout=60):
        """
        Setup all relevant parameters and create scenario
        and instantiate scenario manager
        """
        self._world = world
        self._map = CarlaDataProvider.get_map()
        self.timeout = timeout

        self._trigger_distance = 30
        self._opposite_wait_duration = 5
        self._end_distance = 50

        self._reference_waypoint = self._map.get_waypoint(config.trigger_points[0].location)
        self._construction_wp = None

        self._construction_transforms = []

        self._distance = get_value_parameter(config, 'distance', float, 100)
        self._max_speed = get_value_parameter(config, 'speed', float, 60)
        self._scenario_timeout = 240
        self._direction = get_value_parameter(config, 'direction', str, 'right')
        if self._direction not in ('left', 'right'):
            raise ValueError(f"'direction' must be either 'right' or 'left' but {self._direction} was given")

        super().__init__("ConstructionObstacle", ego_vehicles, config, world, debug_mode, False, criteria_enable)

    def _initialize_actors(self, config):
        """Creates all props part of the construction"""
        self._spawn_side_prop(self._reference_waypoint)

        wps = self._reference_waypoint.next(self._distance)
        if not wps:
            raise ValueError("Couldn't find a viable position to set up the construction actors")
        self._construction_wp = wps[0]
        self._create_construction_setup(self._construction_wp.transform, self._reference_waypoint.lane_width)

        self._end_wp = self._move_waypoint_forward(self._construction_wp, self._end_distance)

    def _move_waypoint_forward(self, wp, distance):
        dist = 0
        next_wp = wp
        while dist < distance:
            next_wps = next_wp.next(1)
            if not next_wps or next_wps[0].is_junction:
                break
            next_wp = next_wps[0]
            dist += 1
        return next_wp

    def _spawn_side_prop(self, wp):
        """Spawn the accident indication signal"""
        prop_wp = wp
        while True:
            if self._direction == "right":
                wp = prop_wp.get_right_lane()
            else:
                wp = prop_wp.get_left_lane()
            if wp is None or wp.lane_type not in (carla.LaneType.Driving, carla.LaneType.Parking):
                break
            prop_wp = wp

        displacement = 0.3 * prop_wp.lane_width
        r_vec = prop_wp.transform.get_right_vector()
        if self._direction == 'left':
            r_vec *= -1

        spawn_transform = wp.transform
        spawn_transform.location += carla.Location(x=displacement * r_vec.x, y=displacement * r_vec.y, z=0.2)
        spawn_transform.rotation.yaw += 90
        signal_prop = CarlaDataProvider.request_new_actor('static.prop.warningconstruction', spawn_transform)
        if not signal_prop:
            raise ValueError("Couldn't spawn the indication prop asset")
        signal_prop.set_simulate_physics(False)
        self.other_actors.append(signal_prop)

    def _create_cones_side(self, start_transform, forward_vector, z_inc=0, cone_length=0, cone_offset=0):
        """Creates the cones at the side"""
        _dist = 0
        while _dist < (cone_length * cone_offset):
            # Move forward
            _dist += cone_offset
            forward_dist = carla.Vector3D(0, 0, 0) + forward_vector * _dist

            location = start_transform.location + forward_dist
            location.z += z_inc
            spawn_transform = carla.Transform(location, start_transform.rotation)
            spawn_transform.location.z -= 200
            cone_transform = carla.Transform(location, start_transform.rotation)

            cone = CarlaDataProvider.request_new_actor('static.prop.constructioncone', spawn_transform)
            cone.set_simulate_physics(False)
            self.other_actors.append(cone)

            self._construction_transforms.append([cone, cone_transform])

    def _create_construction_setup(self, start_transform, lane_width):
        """Create construction setup"""

        _initial_offset = {'cones': {'yaw': 270, 'k': 0.85 * lane_width / 2.0},
                           'warning_sign': {'yaw': 180, 'k': 5, 'z': 0},
                           'debris': {'yaw': 0, 'k': 2, 'z': 1}}
        _prop_names = {'warning_sign': 'static.prop.trafficwarning',
                       'debris': 'static.prop.dirtdebris02'}

        _perp_angle = 90
        _setup = {'lengths': [4, 3], 'offsets': [2, 1]}
        _z_increment = 0.1

        # Traffic warning and debris
        for key, value in _initial_offset.items():
            if key == 'cones':
                continue
            transform = carla.Transform(
                start_transform.location,
                start_transform.rotation)
            transform.rotation.yaw += value['yaw']
            transform.location += value['k'] * \
                transform.rotation.get_forward_vector()
            transform.location.z += value['z']
            transform.rotation.yaw += _perp_angle

            spawn_transform = carla.Transform(transform.location, transform.rotation)
            spawn_transform.location.z -= 200
            static = CarlaDataProvider.request_new_actor(
                _prop_names[key], spawn_transform)
            static.set_simulate_physics(False)
            self.other_actors.append(static)

            self._construction_transforms.append([static, transform])

        # Cones
        side_transform = carla.Transform(
            start_transform.location,
            start_transform.rotation)
        side_transform.rotation.yaw += _perp_angle
        offset_vec = _initial_offset['cones']['k'] * side_transform.rotation.get_forward_vector()
        if self._direction == 'right':
            side_transform.location -= offset_vec
        else:
            side_transform.location += offset_vec

        side_transform.rotation.yaw += _initial_offset['cones']['yaw']

        for i in range(len(_setup['lengths'])):
            self._create_cones_side(
                side_transform,
                forward_vector=side_transform.rotation.get_forward_vector(),
                z_inc=_z_increment,
                cone_length=_setup['lengths'][i],
                cone_offset=_setup['offsets'][i])
            side_transform.location += side_transform.get_forward_vector() * \
                _setup['lengths'][i] * _setup['offsets'][i]
            if i == 0 and self._direction == 'left':
                side_transform.rotation.yaw -= _perp_angle
            else:
                side_transform.rotation.yaw += _perp_angle

    def _create_behavior(self):
        """
        Remove the lane that would collide with the construction and add the construction props.
        Wait until the ego is close to the construction (and a bit more) before changing the side traffic
        Readd the traffic at the end
        """
        root = py_trees.composites.Sequence(name="ConstructionObstacle")
        if self.route_mode:
            root.add_child(RemoveRoadLane(self._reference_waypoint))

        for actor, transform in self._construction_transforms:
            root.add_child(ActorTransformSetter(actor, transform, True))

        end_condition = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        end_condition.add_child(ScenarioTimeout(self._scenario_timeout, self.config.name))
        end_condition.add_child(WaitUntilInFrontPosition(self.ego_vehicles[0], self._end_wp.transform, False))

        behavior = py_trees.composites.Sequence()
        behavior.add_child(InTriggerDistanceToLocation(
            self.ego_vehicles[0], self._construction_wp.transform.location, self._trigger_distance))
        behavior.add_child(Idle(self._opposite_wait_duration))
        if self.route_mode:
            behavior.add_child(SetMaxSpeed(self._max_speed))
        behavior.add_child(WaitForever())

        end_condition.add_child(behavior)
        root.add_child(end_condition)

        if self.route_mode:
            root.add_child(SetMaxSpeed(0))
            root.add_child(ReAddRoadLane(0))
        for actor in self.other_actors:
            root.add_child(ActorDestroy(actor))

        return root

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = [ScenarioTimeoutTest(self.ego_vehicles[0], self.config.name)]
        if not self.route_mode:
            criteria.append(CollisionTest(self.ego_vehicles[0]))
        return criteria

    def __del__(self):
        """
        Remove all actors and traffic lights upon deletion
        """
        self.remove_all_actors()


class ConstructionObstacleTwoWaysLegacy(ConstructionObstacleLegacy):
    """
    Variation of ConstructionObstacle where the ego has to invade the opposite lane
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True, timeout=60):

        self._opposite_interval = get_interval_parameter(config, 'frequency', float, [20, 100])
        super().__init__(world, ego_vehicles, config, randomize, debug_mode, criteria_enable, timeout)

    def _create_behavior(self):
        """
        Remove the lane that would collide with the construction and add the construction props.
        Wait until the ego is close to the construction (and a bit more) before changing the opposite traffic
        Readd the traffic at the end, and allow the ego to invade the opposite lane by deactivating the criteria
        """
        reference_wp = self._construction_wp.get_left_lane()
        if not reference_wp:
            raise ValueError("Couldnt find a left lane to spawn the opposite traffic")

        root = py_trees.composites.Sequence(name="ConstructionObstacleTwoWays")
        if self.route_mode:
            root.add_child(RemoveRoadLane(self._reference_waypoint))

        for actor, transform in self._construction_transforms:
            root.add_child(ActorTransformSetter(actor, transform, True))

        end_condition = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        end_condition.add_child(ScenarioTimeout(self._scenario_timeout, self.config.name))
        end_condition.add_child(WaitUntilInFrontPosition(self.ego_vehicles[0], self._end_wp.transform, False))

        behavior = py_trees.composites.Sequence()
        behavior.add_child(InTriggerDistanceToLocation(
            self.ego_vehicles[0], self._construction_wp.transform.location, self._trigger_distance))
        behavior.add_child(Idle(self._opposite_wait_duration))
        if self.route_mode:
            behavior.add_child(SwitchWrongDirectionTest(False))
            behavior.add_child(ChangeOppositeBehavior(active=False))
            behavior.add_child(OppositeActorFlow(reference_wp, self.ego_vehicles[0], self._opposite_interval))

        end_condition.add_child(behavior)
        root.add_child(end_condition)

        if self.route_mode:
            root.add_child(SwitchWrongDirectionTest(True))
            root.add_child(ChangeOppositeBehavior(active=True))
            root.add_child(ReAddRoadLane(0))
        for actor in self.other_actors:
            root.add_child(ActorDestroy(actor))

        return root


class ParkingExitLegacy(BasicScenario):
    """
    This class holds everything required for a scenario in which the ego would be teleported to the parking lane.
    Once the scenario is triggered, the OutsideRouteLanesTest will be deactivated since the ego is out of the driving lane.
    Then blocking vehicles will be generated in front of and behind the parking point.
    The ego need to exit from the parking lane and then merge into the driving lane.
    After the ego is {end_distance} meters away from the parking point, the OutsideRouteLanesTest will be activated and the scenario ends.

    Note 1: For route mode, this shall be the first scenario of the route. The trigger point shall be the first point of the route waypoints.

    Note 2: Make sure there are enough space for spawning blocking vehicles.
    """

    def __init__(self, world, ego_vehicles, config, debug_mode=False, criteria_enable=True,
                 timeout=180):
        """
        Setup all relevant parameters and create scenario
        and instantiate scenario manager
        """
        self._world = world
        self._map = CarlaDataProvider.get_map()
        self._tm = CarlaDataProvider.get_client().get_trafficmanager(
            CarlaDataProvider.get_traffic_manager_port())
        self.timeout = timeout

        self._bp_attributes = {'base_type': 'car', 'generation': 2}
        self._side_end_distance = 50

        self._front_vehicle_distance = get_value_parameter(config, 'front_vehicle_distance', float, 20)
        self._behind_vehicle_distance = get_value_parameter(config, 'behind_vehicle_distance', float, 10)
        self._direction = get_value_parameter(config, 'direction', str, 'right')
        if self._direction not in ('left', 'right'):
            raise ValueError(f"'direction' must be either 'right' or 'left' but {self._direction} was given")

        self._flow_distance = get_value_parameter(config, 'flow_distance', float, 25)
        self._max_speed = get_value_parameter(config, 'speed', float, 60)
        self._scenario_timeout = 240

        self._end_distance = self._front_vehicle_distance + 15

        # Get parking_waypoint based on trigger_point
        self._trigger_location = config.trigger_points[0].location
        self._reference_waypoint = self._map.get_waypoint(self._trigger_location)
        if self._direction == "left":
            self._parking_waypoint = self._reference_waypoint.get_left_lane()
        else:
            self._parking_waypoint = self._reference_waypoint.get_right_lane()

        if self._parking_waypoint is None:
            raise Exception(
                "Couldn't find parking point on the {} side".format(self._direction))

        super().__init__("ParkingExit",
                         ego_vehicles,
                         config,
                         world,
                         debug_mode,
                         criteria_enable=criteria_enable)

    def _initialize_actors(self, config):
        """
        Custom initialization
        """

        # Spawn the actor in front of the ego
        front_points = self._parking_waypoint.next(
            self._front_vehicle_distance)
        if not front_points:
            raise ValueError("Couldn't find viable position for the vehicle in front of the parking point")

        self.parking_slots.append(front_points[0].transform.location)

        actor_front = CarlaDataProvider.request_new_actor(
            'vehicle.*', front_points[0].transform, rolename='scenario no lights', attribute_filter=self._bp_attributes)
        if actor_front is None:
            raise ValueError("Couldn't spawn the vehicle in front of the parking point")
        actor_front.apply_control(carla.VehicleControl(hand_brake=True))
        self.other_actors.append(actor_front)

        # And move it to the side
        side_location = self._get_displaced_location(actor_front, front_points[0])
        actor_front.set_location(side_location)

        # Spawn the actor behind the ego
        behind_points = self._parking_waypoint.previous(
            self._behind_vehicle_distance)
        if not behind_points:
            raise ValueError("Couldn't find viable position for the vehicle behind the parking point")

        self.parking_slots.append(behind_points[0].transform.location)

        actor_behind = CarlaDataProvider.request_new_actor(
            'vehicle.*', behind_points[0].transform, rolename='scenario no lights', attribute_filter=self._bp_attributes)
        if actor_behind is None:
            actor_front.destroy()
            raise ValueError("Couldn't spawn the vehicle behind the parking point")
        actor_behind.apply_control(carla.VehicleControl(hand_brake=True))
        self.other_actors.append(actor_behind)

        # And move it to the side
        side_location = self._get_displaced_location(actor_behind, behind_points[0])
        actor_behind.set_location(side_location)

        # Move the ego to its side position
        self._ego_location = self._get_displaced_location(self.ego_vehicles[0], self._parking_waypoint)
        self.parking_slots.append(self._ego_location)
        self.ego_vehicles[0].set_location(self._ego_location)

        # Spawn the actor at the side of the ego
        actor_side = CarlaDataProvider.request_new_actor(
            'vehicle.*', self._reference_waypoint.transform, attribute_filter=self._bp_attributes)
        if actor_side is None:
            raise ValueError("Couldn't spawn the vehicle at the side of the parking point")
        self.other_actors.append(actor_side)
        self._tm.update_vehicle_lights(actor_side, True)

        self._end_side_transform = self.ego_vehicles[0].get_transform()
        self._end_side_transform.location.z -= 500

    def _get_displaced_location(self, actor, wp):
        """
        Calculates the transforming such that the actor is at the sidemost part of the lane
        """
        # Move the actor to the edge of the lane near the sidewalk
        displacement = (wp.lane_width - actor.bounding_box.extent.y) / 4
        displacement_vector = wp.transform.get_right_vector()
        if self._direction == 'left':
            displacement_vector *= -1

        new_location = wp.transform.location + carla.Location(x=displacement*displacement_vector.x,
                                                              y=displacement*displacement_vector.y,
                                                              z=displacement*displacement_vector.z)
        new_location.z += 0.05  # Just in case, avoid collisions with the ground
        return new_location

    def _create_behavior(self):
        """
        Deactivate OutsideRouteLanesTest, then move ego to the parking point,
        generate blocking vehicles in front of and behind the ego.
        After ego drives away, activate OutsideRouteLanesTest, end scenario.
        """

        sequence = py_trees.composites.Sequence(name="ParkingExit")
        sequence.add_child(ChangeRoadBehavior(spawn_dist=self._flow_distance))
        root = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        side_actor_behavior = py_trees.composites.Sequence()
        side_actor_behavior.add_child(ChangeAutoPilot(self.other_actors[2], True))
        side_actor_behavior.add_child(DriveDistance(self.other_actors[2], self._side_end_distance))
        side_actor_behavior.add_child(ActorTransformSetter(self.other_actors[2], self._end_side_transform, False))
        side_actor_behavior.add_child(WaitForever())
        root.add_child(side_actor_behavior)

        end_condition = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        end_condition.add_child(DriveDistance(self.ego_vehicles[0], self._end_distance))
        end_condition.add_child(ScenarioTimeout(self._scenario_timeout, self.config.name))
        root.add_child(end_condition)

        sequence.add_child(root)

        for actor in self.other_actors:
            sequence.add_child(ActorDestroy(actor))
        sequence.add_child(ChangeRoadBehavior(spawn_dist=15))

        return sequence

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = [ScenarioTimeoutTest(self.ego_vehicles[0], self.config.name)]
        if not self.route_mode:
            criteria.append(CollisionTest(self.ego_vehicles[0]))
        return criteria

    def __del__(self):
        """
        Remove all actors and traffic lights upon deletion
        """
        self.remove_all_actors()
