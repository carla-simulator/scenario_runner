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
from distutils.log import error

import py_trees
import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import ActorDestroy
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import DriveDistance
from srunner.scenarios.basic_scenario import BasicScenario
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

class Accident(BasicScenario):
    """
    This class holds everything required for a scenario in which there is an accident
    in front of the ego, forcing it to react. A police vehicle is located before
    two other cars that have been in an accident.
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
        self._drive_distance = 120
        self._distance_to_accident = 100
        self._offset = 0.75
        self._first_distance = 10
        self._second_distance = 6
        self._accident_wp = None

        super(Accident, self).__init__("Accident",
                                       ego_vehicles,
                                       config,
                                       world,
                                       randomize,
                                       debug_mode,
                                       criteria_enable=criteria_enable)

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        starting_wp = self._map.get_waypoint(config.trigger_points[0].location)
        accident_wps = starting_wp.next(self._distance_to_accident)
        pre_accident_wps = starting_wp.next(self._distance_to_accident/2)
        if not accident_wps: 
            raise ValueError("Couldn't find a viable position to set up the accident actors")
        if not pre_accident_wps: 
            raise ValueError("Couldn't find a viable position to set up the accident actors")
        self._accident_wp = accident_wps[0]

        # Create the police vehicle
        displacement = self._offset * self._accident_wp.lane_width / 2
        r_vec = self._accident_wp.transform.get_right_vector()
        w_loc = self._accident_wp.transform.location
        w_loc += carla.Location(x=displacement * r_vec.x, y=displacement * r_vec.y)
        police_transform = carla.Transform(w_loc, self._accident_wp.transform.rotation)
        police_car = CarlaDataProvider.request_new_actor('vehicle.dodge.charger_police_2020', police_transform)
        police_lights = carla.VehicleLightState.Special1
        police_lights |= carla.VehicleLightState.Special2
        police_lights |= carla.VehicleLightState.Position
        police_car.set_light_state(carla.VehicleLightState(police_lights))
        self.other_actors.append(police_car)

        # Create the first vehicle that has been in the accident
        vehicle_wps = self._accident_wp.next(self._first_distance)
        if not vehicle_wps: 
            raise ValueError("Couldn't find a viable position to set up the accident actors")
        vehicle_wp = vehicle_wps[0]
        self._accident_wp = pre_accident_wps[0]
        displacement = self._offset * vehicle_wp.lane_width / 2
        r_vec = vehicle_wp.transform.get_right_vector()
        w_loc = vehicle_wp.transform.location
        w_loc += carla.Location(x=displacement * r_vec.x, y=displacement * r_vec.y)
        vehicle_1_transform = carla.Transform(w_loc, vehicle_wp.transform.rotation)
        vehicle_1_car = CarlaDataProvider.request_new_actor('vehicle.tesla.model3', vehicle_1_transform)
        self.other_actors.append(vehicle_1_car)

        # Create the second vehicle that has been in the accident
        vehicle_wps = vehicle_wp.next(self._second_distance)
        if not vehicle_wps: 
            raise ValueError("Couldn't find a viable position to set up the accident actors")
        vehicle_wp = vehicle_wps[0]

        displacement = self._offset * vehicle_wp.lane_width / 2
        r_vec = vehicle_wp.transform.get_right_vector()
        w_loc = vehicle_wp.transform.location
        w_loc += carla.Location(x=displacement * r_vec.x, y=displacement * r_vec.y)
        vehicle_2_transform = carla.Transform(w_loc, vehicle_wp.transform.rotation)
        vehicle_2_car = CarlaDataProvider.request_new_actor('vehicle.mercedes.coupe_2020', vehicle_2_transform)
        self.other_actors.append(vehicle_2_car)

    def _create_behavior(self):
        """
        The vehicle has to drive the whole predetermined distance.
        """
        root = py_trees.composites.Sequence()
        if CarlaDataProvider.get_ego_vehicle_route():
            root.add_child(HandleStartAccidentScenario(self._accident_wp, self._distance_to_accident))
        root.add_child(DriveDistance(self.ego_vehicles[0], self._drive_distance))
        if CarlaDataProvider.get_ego_vehicle_route():
            root.add_child(HandleEndAccidentScenario())
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
