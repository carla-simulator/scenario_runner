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
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import ActorDestroy, BasicAgentBehavior
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import DriveDistance
from srunner.scenarios.basic_scenario import BasicScenario

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

class Accident(BasicScenario):
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
        self.timeout = timeout
        self._drive_distance = 150
        self._distance_to_accident = 50
        self._offset = 0.75

        super(Accident, self).__init__("Accident",
                                       ego_vehicles,
                                       config,
                                       world,
                                       debug_mode,
                                       criteria_enable=criteria_enable)

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        starting_wp = self._map.get_waypoint(config.ego_vehicles[0].transform.location)
        accident_wps = starting_wp.next(self._distance_to_accident)
        if not accident_wps: 
            raise ValueError("Couldn't find a viable position to set up the accident actors")
        accident_wp = accident_wps[0]

        # Part 1, create the parked
        displacement = self._offset * accident_wp.lane_width / 2
        r_vec = accident_wp.transform.get_right_vector()
        w_loc = accident_wp.transform.location
        w_loc += carla.Location(x=displacement * r_vec.x, y=displacement * r_vec.y)
        police_transform = carla.Transform(w_loc, accident_wp.transform.rotation)
        police_car = CarlaDataProvider.request_new_actor('vehicle.dodge.charger_police_2020', police_transform)
        police_lights = carla.VehicleLightState.Special1
        police_lights |= carla.VehicleLightState.Special2
        police_lights |= carla.VehicleLightState.Position
        police_car.set_light_state(carla.VehicleLightState(police_lights))
        self.other_actors.append(police_car)

        # Part 2, create the accident vehicle #1
        accident_wps = accident_wp.next(10)
        if not accident_wps: 
            raise ValueError("Couldn't find a viable position to set up the accident actors")
        accident_wp = accident_wps[0]

        displacement = self._offset * accident_wp.lane_width / 2
        r_vec = accident_wp.transform.get_right_vector()
        w_loc = accident_wp.transform.location
        w_loc += carla.Location(x=displacement * r_vec.x, y=displacement * r_vec.y)
        vehicle_1_transform = carla.Transform(w_loc, accident_wp.transform.rotation)
        vehicle_1_car = CarlaDataProvider.request_new_actor('vehicle.tesla.model3', vehicle_1_transform)
        self.other_actors.append(vehicle_1_car)

        # Part 3, create the accident vehicle #2
        accident_wps = accident_wp.next(6)
        if not accident_wps: 
            raise ValueError("Couldn't find a viable position to set up the accident actors")
        accident_wp = accident_wps[0]

        displacement = self._offset * accident_wp.lane_width / 2
        r_vec = accident_wp.transform.get_right_vector()
        w_loc = accident_wp.transform.location
        w_loc += carla.Location(x=displacement * r_vec.x, y=displacement * r_vec.y)
        vehicle_2_transform = carla.Transform(w_loc, accident_wp.transform.rotation)
        vehicle_2_car = CarlaDataProvider.request_new_actor('vehicle.mercedes.coupe_2020', vehicle_2_transform)
        self.other_actors.append(vehicle_2_car)

    def _create_behavior(self):
        """
        Hero vehicle is entering a junction in an urban area, at a signalized intersection,
        while another actor runs a red lift, forcing the ego to break.
        """
        root = py_trees.composites.Sequence()
        root.add_child(DriveDistance(self.ego_vehicles[0], self._drive_distance))
        root.add_child(ActorDestroy(self.other_actors[0]))
        root.add_child(ActorDestroy(self.other_actors[1]))
        root.add_child(ActorDestroy(self.other_actors[2]))
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
