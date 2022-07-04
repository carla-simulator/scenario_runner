#!/usr/bin/env python
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Object crash without prior vehicle action scenario:
The scenario realizes the user controlled ego vehicle
moving along the road and encountering a construction setup.
"""

from __future__ import print_function

import py_trees
import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorDestroy,
                                                                      ActorTransformSetter,
                                                                      SwitchWrongDirectionTest)
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import DriveDistance
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.background_manager import (ChangeOppositeBehavior,
                                              RemoveRoadLane,
                                              ReAddEgoRoadLane)


class ConstructionObstacle(BasicScenario):
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
        if 'distance' in config.other_parameters:
            self._distance = int(config.other_parameters['distance']['value'])
        else:
            self._distance = 100

        self._takeover_max_dist = 60
        self._drive_distance = self._distance + self._takeover_max_dist

        self._reference_waypoint = self._map.get_waypoint(config.trigger_points[0].location)
        self._construction_wp = None

        self._construction_transforms = []

        super().__init__("ConstructionObstacle", ego_vehicles, config, world, debug_mode, False, criteria_enable)

    def _initialize_actors(self, config):
        """Creates all props part of the construction"""
        wps = self._reference_waypoint.next(self._distance)
        if not wps: 
            raise ValueError("Couldn't find a viable position to set up the construction actors")
        construction_wp = wps[0]
        self._create_construction_setup(construction_wp.transform, self._reference_waypoint.lane_width)

    def create_cones_side(self, start_transform, forward_vector, z_inc=0, cone_length=0, cone_offset=0):
        """Creates the cones at tthe side"""
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

        _initial_offset = {'cones': {'yaw': 180, 'k': lane_width / 2.0},
                           'warning_sign': {'yaw': 180, 'k': 5, 'z': 0},
                           'debris': {'yaw': 0, 'k': 2, 'z': 1}}
        _prop_names = {'warning_sign': 'static.prop.trafficwarning',
                       'debris': 'static.prop.dirtdebris02'}

        _perp_angle = 90
        _setup = {'lengths': [0, 6, 3], 'offsets': [0, 2, 1]}
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
        side_transform.location -= _initial_offset['cones']['k'] * \
            side_transform.rotation.get_forward_vector()
        side_transform.rotation.yaw += _initial_offset['cones']['yaw']

        for i in range(len(_setup['lengths'])):
            self.create_cones_side(
                side_transform,
                forward_vector=side_transform.rotation.get_forward_vector(),
                z_inc=_z_increment,
                cone_length=_setup['lengths'][i],
                cone_offset=_setup['offsets'][i])
            side_transform.location += side_transform.get_forward_vector() * \
                _setup['lengths'][i] * _setup['offsets'][i]
            side_transform.rotation.yaw += _perp_angle

    def _create_behavior(self):
        """
        Only behavior here is to wait
        """
        root = py_trees.composites.Sequence()

        for actor, transform in self._construction_transforms:
            root.add_child(ActorTransformSetter(actor, transform, True))

        root.add_child(DriveDistance(self.ego_vehicles[0], self._drive_distance))
        for i, _ in enumerate(self.other_actors):
            root.add_child(ActorDestroy(self.other_actors[i]))

        if not self.route_mode:
            return root

        sequence = py_trees.composites.Sequence()
        sequence.add_child(RemoveRoadLane(self._reference_waypoint))
        sequence.add_child(root)
        sequence.add_child(ReAddEgoRoadLane())
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
        Remove all actors and traffic lights upon deletion
        """
        self.remove_all_actors()


class ConstructionObstacleTwoWays(ConstructionObstacle):
    """
    Variation of ConstructionObstacle where the ego has to invade the opposite lane
    """
    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True, timeout=60):

        if 'frequency' in config.other_parameters:
            self._opposite_frequency = float(config.other_parameters['frequency']['value'])
        else:
            self._opposite_frequency = 200
        super().__init__(world, ego_vehicles, config, randomize, debug_mode, criteria_enable, timeout)

    def _create_behavior(self):
        """
        Only behavior here is to wait
        """
        root = py_trees.composites.Sequence()
        for actor, transform in self._construction_transforms:
            root.add_child(ActorTransformSetter(actor, transform, True))
        root.add_child(DriveDistance(self.ego_vehicles[0], self._drive_distance))

        for i, _ in enumerate(self.other_actors):
            root.add_child(ActorDestroy(self.other_actors[i]))

        if not self.route_mode:
            return root

        sequence = py_trees.composites.Sequence()
        sequence.add_child(SwitchWrongDirectionTest(False))
        sequence.add_child(ChangeOppositeBehavior(spawn_dist=self._opposite_frequency))
        sequence.add_child(RemoveRoadLane(self._reference_waypoint))
        sequence.add_child(root)
        sequence.add_child(SwitchWrongDirectionTest(True))
        sequence.add_child(ChangeOppositeBehavior(spawn_dist=50))
        sequence.add_child(ReAddEgoRoadLane())
        return sequence
