#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Actor helpers.

Location and transform come straight from the Python API; only the speed needs
a helper, because it is the ground speed and deliberately ignores the z axis.
Destruction is here too, as every actor is destroyed by whoever created it,
and so are the lookups that used to walk the actor pool.
"""

import math

# An actor has no transform until the world advances once after it was spawned,
# and reports this until then. Spawning with tick=False leaves it in that state.
UNTICKED_LOCATION = (0.0, 0.0, 0.0)


def has_actor_ticked(actor):
    """
    Whether the actor's state is readable yet.

    An actor spawned with tick=False has no transform until the world next
    advances, and reports the origin meanwhile. No road position is the origin,
    so that is what distinguishes the two.
    """
    location = actor.get_location()
    return (location.x, location.y, location.z) != UNTICKED_LOCATION


def get_ground_speed(actor):
    """Returns the actor's ground speed in m/s, ignoring any vertical motion"""
    velocity = actor.get_velocity()
    return math.sqrt(velocity.x ** 2 + velocity.y ** 2)


def destroy_actor(actor):
    """
    Destroys an actor, returning whether it was still there to destroy.

    Whoever creates an actor destroys it, so this is the only place that has to
    tolerate an actor that is already gone.
    """
    if actor is None or not actor.is_alive:
        return False
    return actor.destroy()


def destroy_actors(actors):
    """Destroys every actor of a list, returning how many were still alive"""
    return sum(destroy_actor(actor) for actor in actors)


def find_actor_by_name(world, role_name):
    """
    Returns the first actor of the world carrying that role name, or None.

    The world's list is the only one both complete and current, so an actor
    another process spawned is found too.
    """
    for actor in world.get_actors():
        if actor.attributes.get('role_name') == role_name:
            return actor
    print("Non-existing actor name {}".format(role_name))
    return None


def get_hero_actor(world):
    """Returns the ego vehicle, i.e. the actor whose role name is 'hero'"""
    return find_actor_by_name(world, 'hero')
