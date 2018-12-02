# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides implementation for GlobalRoutePlannerDAO
"""

import carla


class GlobalRoutePlannerDAO(object):
    """
    This class is the data access layer for fetching data
    from the carla server instance for GlobalRoutePlanner
    """

    def __init__(self, wmap):
        """ Constructor """
        self.wmap = wmap

    def get_topology(self):
        """ Accessor for topology """
        topology = []
        # Transforming topology into list of vertex pairs
        for segment in self.wmap.get_topology():
            x1 = segment[0].transform.location.x
            y1 = segment[0].transform.location.y
            x2 = segment[1].transform.location.x
            y2 = segment[1].transform.location.y
            topology.append([(x1, y1), (x2, y2)])
        return topology

    def get_next_waypoint(self, location, distance):
        """ Accessor for wayponit """
        x, y = location
        location = carla.Location(x=x, y=y, z=0.0)
        waypoint = self.wmap.get_waypoint(location)
        nxt = list(waypoint.next(distance))[0].transform.location
        return nxt.x, nxt.y
