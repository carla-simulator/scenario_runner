#!/usr/bin/env python

# Copyright (c) 2018-2019 Intel Labs.
# authors: Olaf Duenkel (olaf.dunkel@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provide the basic class for all user-defined scenarios.
"""

from __future__ import print_function
import math
import numpy as np
import carla


def get_crossing_point(ego_actor):
    """
    Get the next crossing point location in front of the ego vehicle
    
    @return point of crossing
    """
    wp_cross = ego_actor.get_world().get_map().get_waypoint(ego_actor.get_location())
    
    while( not wp_cross.is_intersection ):
        wp_cross = wp_cross.next(2)[0]
        
    crossing = carla.Location(x=wp_cross.transform.location.x, y=wp_cross.transform.location.y, z=wp_cross.transform.location.z)    
    
    return crossing

    
def get_geometric_linear_intersection(ego_actor, other_actor):
    """
    Obtain a intersection point between two actor's location by using their waypoints (wp)

    @return point of intersection of the two vehicles
    """
    
    wp_ego_1 = ego_actor.get_world().get_map().get_waypoint(ego_actor.get_location())
    wp_ego_2 = wp_ego_1.next(1)[0]
    x_ego_1 = wp_ego_1.transform.location.x
    y_ego_1 = wp_ego_1.transform.location.y
    x_ego_2 = wp_ego_2.transform.location.x
    y_ego_2 = wp_ego_2.transform.location.y
    
    wp_other_1 = other_actor.get_world().get_map().get_waypoint(other_actor.get_location())
    wp_other_2 = wp_other_1.next(1)[0]
    x_other_1 = wp_other_1.transform.location.x
    y_other_1 = wp_other_1.transform.location.y
    x_other_2 = wp_other_2.transform.location.x
    y_other_2 = wp_other_2.transform.location.y
    
    s = np.vstack( [ (x_ego_1, y_ego_1), (x_ego_2, y_ego_2), (x_other_1, y_other_1), (x_other_2, y_other_2) ] )
    h = np.hstack( (s, np.ones((4,1))) )
    line1 = np.cross( h[0], h[1] )
    line2 = np.cross( h[2], h[3] )
    x,y,z = np.cross( line1, line2 )
    if z == 0:
        return (float('inf'), float('inf'))
    
    intersection = carla.Location(x=x/z, y=y/z, z=0)
    
    return intersection