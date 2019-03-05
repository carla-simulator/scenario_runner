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

import carla


def calculate_velocity(actor):
    """
    Method to calculate the velocity of a actor
    """
    velocity_squared = actor.get_velocity().x**2
    velocity_squared += actor.get_velocity().y**2
    return math.sqrt(velocity_squared)


class CarlaDataProvider(object):

    """
    This class provides access to various data of all registered actors
    It buffers the data and updates it on every CARLA tick

    Currently available data:
    - Absolute velocity
    - Location

    Potential additions:
    - Acceleration
    - Transform

    In addition it provides access to the map and the transform of all traffic lights
    """

    _actor_velocity_map = dict()
    _actor_location_map = dict()
    _traffic_light_map = dict()
    _map = None

    @staticmethod
    def register_actor(actor):
        """
        Add new actor to dictionaries
        If actor already exists, throw an exception
        """
        if actor in CarlaDataProvider._actor_velocity_map:
            raise KeyError(
                "Vehicle '{}' already registered. Cannot register twice!".format(actor))
        else:
            CarlaDataProvider._actor_velocity_map[actor] = 0.0

        if actor in CarlaDataProvider._actor_location_map:
            raise KeyError(
                "Vehicle '{}' already registered. Cannot register twice!".format(actor.id))
        else:
            CarlaDataProvider._actor_location_map[actor] = None

    @staticmethod
    def register_actors(actors):
        """
        Add new set of actors to dictionaries
        """
        for actor in actors:
            CarlaDataProvider.register_actor(actor)

    @staticmethod
    def on_carla_tick():
        """
        Callback from CARLA
        """
        for actor in CarlaDataProvider._actor_velocity_map:
            if actor is not None and actor.is_alive:
                CarlaDataProvider._actor_velocity_map[actor] = calculate_velocity(actor)

        for actor in CarlaDataProvider._actor_location_map:
            if actor is not None and actor.is_alive:
                CarlaDataProvider._actor_location_map[actor] = actor.get_location()

    @staticmethod
    def get_velocity(actor):
        """
        returns the absolute velocity for the given actor
        """
        if actor not in CarlaDataProvider._actor_velocity_map.keys():
            # We are initentionally not throwing here
            # This may cause exception loops in py_trees
            return 0.0
        else:
            return CarlaDataProvider._actor_velocity_map[actor]

    @staticmethod
    def get_location(actor):
        """
        returns the location for the given actor
        """
        if actor not in CarlaDataProvider._actor_location_map.keys():
            # We are initentionally not throwing here
            # This may cause exception loops in py_trees
            return None
        else:
            return CarlaDataProvider._actor_location_map[actor]

    @staticmethod
    def prepare_map(actor):
        """
        This function set the current map and loads all traffic lights for this map to
        _traffic_light_map
        """

        if CarlaDataProvider._map is None:
            CarlaDataProvider._map = actor.get_world().get_map()

            # Parse all traffic lights
            CarlaDataProvider._traffic_light_map.clear()
            for traffic_light in actor.get_world().get_actors().filter('*traffic_light*'):
                if traffic_light not in CarlaDataProvider._traffic_light_map.keys():
                    CarlaDataProvider._traffic_light_map[traffic_light] = traffic_light.get_transform()
                else:
                    raise KeyError(
                        "Traffic light '{}' already registered. Cannot register twice!".format(traffic_light.id))

    @staticmethod
    def get_next_traffic_light(actor, use_cached_location=True):
        """
        returns the next relevant traffic light for the provided actor
        """
        CarlaDataProvider.prepare_map(actor)
        location = CarlaDataProvider.get_location(actor)
        if not use_cached_location:
            location = actor.get_transform().location

        waypoint = CarlaDataProvider._map.get_waypoint(location)

        # Create list of all waypoints until next intersection
        list_of_waypoints = []
        while not waypoint.is_intersection:
            list_of_waypoints.append(waypoint)
            waypoint = waypoint.next(2.0)[0]

        # If the list is empty, the actor is in an intersection
        if not list_of_waypoints:
            return None

        relevant_traffic_light = None
        distance_to_relevant_traffic_light = float("inf")

        for traffic_light in CarlaDataProvider._traffic_light_map:
            if hasattr(traffic_light, 'trigger_volume'):
                tl_t = CarlaDataProvider._traffic_light_map[traffic_light]
                transformed_tv = tl_t.transform(traffic_light.trigger_volume.location)
                distance = carla.Location(transformed_tv).distance(list_of_waypoints[-1].transform.location)

                if distance < distance_to_relevant_traffic_light:
                    relevant_traffic_light = traffic_light
                    distance_to_relevant_traffic_light = distance

        return relevant_traffic_light

    @staticmethod
    def cleanup():
        """
        Cleanup and remove all entries from all dictionaries
        """
        CarlaDataProvider._actor_velocity_map.clear()
        CarlaDataProvider._actor_location_map.clear()
        CarlaDataProvider._traffic_light_map.clear()
        CarlaDataProvider._map = None
