#!/usr/bin/env python

# Copyright (c) 2018-2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides all frequently used data from CARLA via
local buffers to avoid blocking calls to CARLA
"""

import math
import random

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
    _world = None
    _sync_flag = False

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
    def prepare_map():
        """
        This function set the current map and loads all traffic lights for this map to
        _traffic_light_map
        """
        if CarlaDataProvider._map is None:
            CarlaDataProvider._map = CarlaDataProvider._world.get_map()

        # Parse all traffic lights
        CarlaDataProvider._traffic_light_map.clear()
        for traffic_light in CarlaDataProvider._world.get_actors().filter('*traffic_light*'):
            if traffic_light not in CarlaDataProvider._traffic_light_map.keys():
                CarlaDataProvider._traffic_light_map[traffic_light] = traffic_light.get_transform()
            else:
                raise KeyError(
                    "Traffic light '{}' already registered. Cannot register twice!".format(traffic_light.id))

    @staticmethod
    def get_world():
        return CarlaDataProvider._world

    @staticmethod
    def is_sync_mode():
        return CarlaDataProvider._sync_flag

    @staticmethod
    def set_world(world):
        CarlaDataProvider._world = world
        settings = world.get_settings()
        CarlaDataProvider._sync_flag = settings.synchronous_mode
        CarlaDataProvider._map = CarlaDataProvider._world.get_map()

    @staticmethod
    def get_map(world=None):
        """
        Get the current map
        """
        if CarlaDataProvider._map is None:
            if world is None:
                if CarlaDataProvider._world is None:
                    raise ValueError("class member \'world'\' not initialized yet")
                else:
                    CarlaDataProvider._map = CarlaDataProvider._world.get_map()
            else:
                CarlaDataProvider._map = world.get_map()

        return CarlaDataProvider._map

    @staticmethod
    def get_next_traffic_light(actor, use_cached_location=True):
        """
        returns the next relevant traffic light for the provided actor
        """

        CarlaDataProvider.prepare_map()
        location = CarlaDataProvider.get_location(actor)

        if not use_cached_location:
            location = actor.get_transform().location

        waypoint = CarlaDataProvider._map.get_waypoint(location)
        # Create list of all waypoints until next intersection
        list_of_waypoints = []
        while waypoint and not waypoint.is_intersection:
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


class CarlaActorPool(object):

    """
    The CarlaActorPool caches all scenario relevant actors.
    It works similar to a singelton.

    An actor can be created via "request_actor", and access
    is possible via "get_actor_by_id".

    Using CarlaActorPool, actors can be shared between scenarios.
    """

    _world = None
    _carla_actor_pool = dict()
    _spawn_points = None

    @staticmethod
    def set_world(world):
        """
        Set the CARLA world
        """
        CarlaActorPool._world = world
        CarlaActorPool.generate_spawn_points()

    @staticmethod
    def generate_spawn_points():
        spawn_points = list(CarlaDataProvider.get_map(CarlaActorPool._world).get_spawn_points())
        random.shuffle(spawn_points)
        CarlaActorPool._spawn_points = spawn_points

    @staticmethod
    def setup_actor(model, spawn_point, hero=False, autopilot=False, random_location=False):
        """
        Function to setup the most relevant actor parameters,
        incl. spawn point and vehicle model.
        """

        blueprint_library = CarlaActorPool._world.get_blueprint_library()

        # Get vehicle by model
        blueprint = random.choice(blueprint_library.filter(model))
        if hero:
            blueprint.set_attribute('role_name', 'hero')
        else:
            blueprint.set_attribute('role_name', 'scenario')

        if random_location:
            actor = None
            while not actor:
                spawn_point = random.choice(CarlaActorPool._spawn_points)
                actor = CarlaActorPool._world.try_spawn_actor(blueprint, spawn_point)

        else:
            actor = CarlaActorPool._world.try_spawn_actor(blueprint, spawn_point)

        if actor is None:
            raise Exception(
                "Error: Unable to spawn vehicle {} at {}".format(model, spawn_point))
        else:
            # Let's deactivate the autopilot of the actor if it belongs to vehicle
            if actor in blueprint_library.filter('vehicle.*'):
                actor.set_autopilot(autopilot)
            else:
                pass
        # wait for the actor to be spawned properly before we do anything
        CarlaActorPool._world.tick()
        CarlaActorPool._world.wait_for_tick()
        return actor

    @staticmethod
    def request_new_actor(model, spawn_point, hero=False, autopilot=False, random_location=False):
        """
        This method tries to create a new actor. If this was
        successful, the new actor is returned, None otherwise.
        """
        actor = CarlaActorPool.setup_actor(model, spawn_point, hero, autopilot, random_location)

        if actor is not None:
            CarlaActorPool._carla_actor_pool[actor.id] = actor
            return actor
        else:
            return None

    @staticmethod
    def get_actor_by_id(actor_id):
        """
        Get an actor from the pool by using its ID. If the actor
        does not exist, None is returned.
        """
        if actor_id in CarlaActorPool._carla_actor_pool:
            return CarlaActorPool._carla_actor_pool[actor_id]
        else:
            print("Non-existing actor id {}".format(actor_id))
            return None

    @staticmethod
    def remove_actor_by_id(actor_id):
        """
        Remove an actor from the pool using its ID
        """
        if actor_id in CarlaActorPool._carla_actor_pool:
            CarlaActorPool._carla_actor_pool[actor_id].destroy()
            CarlaActorPool._carla_actor_pool[actor_id] = None
            CarlaActorPool._carla_actor_pool.pop(actor_id)
        else:
            print("Trying to remove a non-existing actor id {}".format(actor_id))

    @staticmethod
    def cleanup():
        """
        Cleanup the actor pool, i.e. remove and destroy all actors
        """
        for actor_id in CarlaActorPool._carla_actor_pool.copy():
            CarlaActorPool._carla_actor_pool[actor_id].destroy()
            CarlaActorPool._carla_actor_pool.pop(actor_id)

        CarlaActorPool._carla_actor_pool = dict()
        CarlaActorPool._world = None

    @staticmethod
    def remove_all_actors_in_surrounding(location, distance):
        """
        Remove all actors from the pool that are closer than distance to the
        provided location
        """
        for actor_id in CarlaActorPool._carla_actor_pool.copy():
            if CarlaActorPool._carla_actor_pool[actor_id].get_location().distance(location) < distance:
                CarlaActorPool._carla_actor_pool[actor_id].destroy()
                CarlaActorPool._carla_actor_pool.pop(actor_id)

        # Remove all keys with None values
        CarlaActorPool._carla_actor_pool = dict({k: v for k, v in CarlaActorPool._carla_actor_pool.items() if v})
