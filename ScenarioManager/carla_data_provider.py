#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: Fabian Oboril (fabian.oboril@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides all frequently used data from CARLA via
local buffers to avoid blocking calls to CARLA
"""

import math


def calculate_velocity(vehicle):
    """
    Method to calculate the velocity of a vehicle
    """
    velocity_squared = vehicle.get_velocity().x**2
    velocity_squared += vehicle.get_velocity().y**2
    return math.sqrt(velocity_squared)


class CarlaDataProvider(object):

    """
    This class provides access to various data of all registered vehicles
    It buffers the data and updates it on every CARLA tick

    Currently available data:
    - Absolute velocity
    - Location

    Potential additions:
    - Acceleration
    - Transform
    """

    _vehicle_velocity_map = dict()
    _vehicle_location_map = dict()

    @staticmethod
    def register_vehicle(vehicle):
        """
        Add new vehicle to dictionaries
        If vehicle already exists, throw an exception
        """
        if vehicle in CarlaDataProvider._vehicle_velocity_map:
            raise KeyError(
                "Vehicle '{}' already registered. Cannot register twice!".format(vehicle))
        else:
            CarlaDataProvider._vehicle_velocity_map[vehicle] = 0.0

        if vehicle in CarlaDataProvider._vehicle_location_map:
            raise KeyError(
                "Vehicle '{}' already registered. Cannot register twice!".format(vehicle.id))
        else:
            CarlaDataProvider._vehicle_location_map[vehicle] = None

    @staticmethod
    def register_vehicles(vehicles):
        """
        Add new set of vehicles to dictionaries
        """
        for vehicle in vehicles:
            CarlaDataProvider.register_vehicle(vehicle)

    @staticmethod
    def on_carla_tick():
        """
        Callback from CARLA
        """
        for vehicle in CarlaDataProvider._vehicle_velocity_map:
            if vehicle is not None and vehicle.is_alive:
                CarlaDataProvider._vehicle_velocity_map[
                    vehicle] = calculate_velocity(vehicle)

        for vehicle in CarlaDataProvider._vehicle_location_map:
            if vehicle is not None and vehicle.is_alive:
                CarlaDataProvider._vehicle_location_map[
                    vehicle] = vehicle.get_location()

    @staticmethod
    def get_velocity(vehicle):
        """
        returns the absolute velocity for the given vehicle
        """
        if vehicle not in CarlaDataProvider._vehicle_velocity_map.keys():
            # We are initentionally not throwing here
            # This may cause exception loops in py_trees
            return 0.0
        else:
            return CarlaDataProvider._vehicle_velocity_map[vehicle]

    @staticmethod
    def get_location(vehicle):
        """
        returns the location for the given vehicle
        """
        if vehicle not in CarlaDataProvider._vehicle_location_map.keys():
            # We are initentionally not throwing here
            # This may cause exception loops in py_trees
            return None
        else:
            return CarlaDataProvider._vehicle_location_map[vehicle]

    @staticmethod
    def cleanup():
        """
        Cleanup and remove all entries from all dictionaries
        """
        CarlaDataProvider._vehicle_velocity_map.clear()
        CarlaDataProvider._vehicle_location_map.clear()
