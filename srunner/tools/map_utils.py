#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Helpers around the map name, which CARLA UE5 is stricter about than 0.9.x was.
"""


def resolve_map_name(client, town):
    # type: (object, str) -> str
    """
    Resolve a town name against the maps available on the server. CARLA UE5
    requires exact names and ships the classic towns as layered variants
    (e.g. 'Town01' only exists as 'Town01_Opt').
    """
    if client is None:
        return town
    try:
        available = [m.split('/')[-1] for m in client.get_available_maps()]
    except RuntimeError:
        return town
    if town in available:
        return town
    for candidate in ('{}_Opt'.format(town), '{}HD_Opt'.format(town), '{}HD'.format(town)):
        if candidate in available:
            print("WARNING: Map {} does not exist in this CARLA version. Using {} instead".format(
                town, candidate))
            return candidate
    return town
