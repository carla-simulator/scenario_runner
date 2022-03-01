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
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import ActorDestroy
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import DriveDistance
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import Idle
from srunner.scenarios.object_crash_vehicle import StationaryObjectCrossing
from srunner.tools.background_manager import HandleStartAccidentScenario, HandleEndAccidentScenario

def convert_dict_to_transform(actor_dict):
    """
    Convert a JSON string to a CARLA transform
    """
    return carla.Transform(
        carla.Location(
            x=float(actor_dict['x']),
            y=float(actor_dict['y']),
            z=float(actor_dict['z'])
        ),
        carla.Rotation(
            roll=0.0,
            pitch=0.0,
            yaw=float(actor_dict['yaw'])
        )
    )

class ConstructionSetupCrossing(StationaryObjectCrossing):

    """
    This class holds everything required for a construction scenario
    The ego vehicle is passing through a road and encounters
    a stationary rectangular construction cones setup and traffic warning.

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
        self._drive_distance = 150
        self._distance_to_construction = 100
        self.construction_wp = None

        super(ConstructionSetupCrossing, self).__init__(world,
                                                        ego_vehicles,
                                                        config,
                                                        randomize,
                                                        debug_mode,
                                                        criteria_enable)

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        lane_width = self._reference_waypoint.lane_width
        starting_wp = self._map.get_waypoint(config.trigger_points[0].location)
        construction_wps = starting_wp.next(self._distance_to_construction)
        if not construction_wps: 
            raise ValueError("Couldn't find a viable position to set up the accident actors")
        self.construction_wp = construction_wps[0]

        self._create_construction_setup(self.construction_wp.transform, lane_width)
        pre_accident_wps = starting_wp.next(self._distance_to_construction/2)
        if not construction_wps: 
            raise ValueError("Couldn't find a viable position to set up the accident actors")
        self.construction_wp = pre_accident_wps[0]

    def create_cones_side(self, start_transform, forward_vector, z_inc=0, 
                          cone_length=0, cone_offset=0):
        
        _dist = 0
        while _dist < (cone_length * cone_offset):
            # Move forward
            _dist += cone_offset
            forward_dist = carla.Vector3D(0, 0, 0) + forward_vector * _dist

            location = start_transform.location + forward_dist
            location.z += z_inc
            transform = carla.Transform(location, start_transform.rotation)

            cone = CarlaDataProvider.request_new_actor(
                'static.prop.constructioncone', transform)
            cone.set_simulate_physics(True)
            self.other_actors.append(cone)

    def _create_construction_setup(self, start_transform, lane_width):
        """
        Create Construction Setup
        """

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
            static = CarlaDataProvider.request_new_actor(
                _prop_names[key], transform)
            static.set_simulate_physics(True)
            self.other_actors.append(static)

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
        if CarlaDataProvider.get_ego_vehicle_route():
            root.add_child(HandleStartAccidentScenario(self.construction_wp, self._distance_to_construction))
        root.add_child(DriveDistance(self.ego_vehicles[0], self._drive_distance))
        if CarlaDataProvider.get_ego_vehicle_route():
            root.add_child(HandleEndAccidentScenario())
        root.add_child(Idle(15))
        for i, _ in enumerate(self.other_actors):
            root.add_child(ActorDestroy(self.other_actors[i]))
        return root


    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        return [CollisionTest(self.ego_vehicles[0])]

    def __del__(self):
        """
        Remove all actors and traffic lights upon deletion
        """
        self.remove_all_actors()
