# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides GlobalRoutePlanner implementation.
"""

import math
import networkx as nx
import carla


class GlobalRoutePlanner(object):
    """
    This class provides a very high level route plan.
    Instantiate the calss by passing a reference to
    A GlobalRoutePlannerDAO object.
    """

    def __init__(self, dao):
        """
        Constructor
        """
        self.dao = dao

    def setup(self):
        """
        Perform initial server data lookup and builds graph representation
        of the world map.
        """
        self.topology = self.dao.get_topology()
        # Creating graph of the world map and also a map from
        # node co-ordinates to node id
        self.graph, self.id_map = self.build_graph()

    def plan_route(self, origin, heading, destination, graph, idmap, topology):
        """
        The following function generates the route plan based on
        origin      : tuple containing x, y of the route's start position
        destination : tuple containing x, y of the route's end position
        heading     : current heading of the vehicle in radian

        return      : list of turn by turn navigation decision
        possible values (for now) are START, GO_STRAIGHT, LEFT, RIGHT,
        FOLLOW_LANE, STOP
        """
        xo, yo = origin
        xd, yd = destination

        start = self.localise(xo, yo, topology)
        end = self.localise(xd, yd, topology)
        start = self.align(start, self.get_direction(start))
        end = self.align(end, self.get_direction(end))

        route = nx.shortest_path(graph, source=idmap[start[0]],
                                 target=idmap[end[1]],
                                 weight='distance')

        plan = []
        plan.append('START')
        for i in [x for x in range(len(route)-2) if x % 2 == 0]:
            v1 = self.unit_vector(graph.nodes[route[i]]['vertex'],
                                  graph.nodes[route[i+1]]['vertex'])
            v2 = self.unit_vector(graph.nodes[route[i+1]]['vertex'],
                                  graph.nodes[route[i+2]]['vertex'])
            direction = math.atan2(*v2[::-1]) - math.atan2(*v1[::-1])
            if abs(direction) < 0.174533:
                plan.append('GO_STRAIGHT')
            elif direction > 0:
                plan.append('LEFT')
            elif direction < 0:
                plan.append('RIGHT')
        plan.append('STOP')

        return plan

    def build_graph(self):
        """
        This function builds a graph representation of topology
        """
        graph = nx.DiGraph()
        # map with structure {(x,y): id, ... }
        id_map = dict()

        for segment in self.topology:

            direction = self.get_direction(segment)
            segment = self.align(segment, direction)
            for vertex in segment:
                if vertex not in id_map:
                    new_id = len(id_map)
                    id_map[vertex] = new_id
                    graph.add_node(new_id, vertex=vertex)
            p1, p2 = segment
            n1, n2 = id_map[p1], id_map[p2]
            graph.add_edge(n1, n2, distance=self.distance(p1, p2))

        return graph, id_map

    def align(self, segment, vector):
        """
        This function returns the segment with its vertex order
        aligned along vector input
        """
        direction = self.get_direction(segment)
        p1, p2 = segment
        midpoint = ((p1[0]+p2[0])/2.0, (p1[1]+p2[1])/2.0)
        v1 = self.unit_vector(midpoint, p1)
        v2 = self.unit_vector(midpoint, p2)
        if self.dot(direction, v2) < self.dot(direction, v1):
                segment = (p2, p1)

        return segment

    def get_direction(self, segment):
        """
        This function returns a unit vector along the allowed direction
        of travel in the road segment
        """
        p1, p2 = segment
        midpoint = ((p1[0]+p2[0])/2.0, (p1[1]+p2[1])/2.0)
        segment_length = self.distance(p1, p2)
        nxt_waypoint = self.dao.get_next_waypoint(midpoint, 0.1*segment_length)
        return self.unit_vector(midpoint, nxt_waypoint)

    def localise(self, x, y, topology):
        """
        This function finds the road segment closest to (x, y)
        """
        distance = float('inf')
        nearest_segment = (distance, (float('inf'), float('inf')))
        # Finding the road segment with the least distance from (x, y)
        # and also such that (x, y) lies inside the circle formed by the
        # segment as diameter
        for segment in topology:
            distance = self.distance_to_line(segment[0],
                                             segment[1], (x, y))
            v1 = self.unit_vector((x, y), segment[0])
            v2 = self.unit_vector((x, y), segment[1])
            if self.dot(v1, v2) < 0 and distance < nearest_segment[0]:
                nearest_segment = (distance, segment)
        segment = nearest_segment[1]

        return segment

    def distance(self, point1, point2):
        """
        returns the distance between point1 and point2
        """
        x1, y1 = point1
        x2, y2 = point2
        return math.sqrt((x2-x1)**2+(y2-y1)**2)

    def distance_to_line(self, point1, point2, target):
        """
        This functions returns the distance between the target point and
        The line joining point1, point2. Accurate to 5 decimal places.
        """
        x1, y1 = point1
        x2, y2 = point2
        xt, yt = target
        m = (y2-y1)/((x2-x1)+0.00001)

        a = m
        b = -1
        c = y1 - m*x1

        return abs(a*xt+b*yt+c)/math.sqrt(a**2+b**2)

    def unit_vector(self, point1, point2):
        """
        This function returns the unit vector from point1 to point2
        """
        x1, y1 = point1
        x2, y2 = point2

        vector = (x2-x1, y2-y1)
        vector_mag = math.sqrt(vector[0]**2+vector[1]**2)
        vector = (vector[0]/vector_mag, vector[1]/vector_mag)

        return vector

    def dot(self, vector1, vector2):
        """
        This function returns the dot product of vector1 with vector2
        """
        return vector1[0]*vector2[0]+vector1[1]*vector2[1]

    pass
