# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# This file contains the class for global route planning

"""
This module provides GlobalRoutePlanner implementation.
"""

import math


class GlobalRoutePlanner(object):

    """
    This class provides a very high level route plan.
    Instantiate the calss by passing a reference to carla world.
    """

    def __init__(self, world):
        """
        Process the topology returned by world into a list of simple
        co-ordinate pairs
        """
        self.world = world
        self.topology = []
        for segment in self.world.get_map().get_topology():
            x1 = segment[0].transform.location.x
            y1 = segment[0].transform.location.y
            x2 = segment[1].transform.location.x
            y2 = segment[1].transform.location.y
            self.topology.append([(x1, y1), (x2, y2)])
        pass
        self.topology = self.roundoff(self.topology)
        self.graph, self.id_map = self.build_graph(self.topology)

    def plan_route(self, origin, destination, heading):
        """
        The following function generates the route plan based on
        origin      : tuple containing x, y of the route's start position
        destination : tuple containing x, y of the route's end position
        heading     : current heading of the vehicle in radian

        return      : list of turn by turn navigation decision
        possible values are GO_STRAIGHT, LEFT, RIGHT,
        CHANGE_LANE_LEFT, CHANGE_LANE_RIGHT
        """

        x_origin, y_origin = origin
        x_destination, y_destination = destination

        x_start, y_start = self.localise(x_origin, y_origin,
                                         heading, self.topology)
        x_end, y_end = self.localise(x_destination, y_destination,
                                     heading, self.topology)

        return None

    def build_graph(self, topology):
        """
        This function builds a graph representation of topology
        """

        # Structure of graph dictionary {id: node, ... }
        graph = dict()
        # Node set with structure {(x,y): id, ... }
        id_map = dict()

        for segment in topology:
            for vertex in segment:
                if vertex not in id_map:
                    new_id = len(id_map)
                    id_map[vertex] = new_id
                    graph[new_id] = self.node(new_id, vertex)
            graph[id_map[segment[0]]].add_connection(id_map[segment[1]])
            graph[id_map[segment[1]]].add_connection(id_map[segment[0]])

        return graph, id_map

    class node(object):
        """
        node object in the topology graph
        """

        def __init__(self, id, vertex):
            self.id = id
            self.vertex = vertex  # vertex co-ordinate pair as a tuple
            self.connections = []   # list of connecting node ids

        def add_connection(self, connecting_node_id):
            self.connections.append(connecting_node_id)
            pass
        pass

    def roundoff(self, topology):
        """
        This function rounds off the co-ordinate values in the topology
        list to 1cm.
        """

        def dist_check(p1, p2, threshold):
            return math.sqrt((p2[0]-p1[0])**2+(p2[1]-p1[1])**2) < threshold

        for i, segment in enumerate(topology):
            for j in range(i+1, len(topology)):

                for vid_self, vertex_self in enumerate(segment):
                    for vid_other, vertex_other in enumerate(topology[j]):

                        if dist_check(vertex_self, vertex_other, 0.01):
                            x = round(segment[vid_self][0], 2)
                            y = round(segment[vid_self][1], 2)
                            try:
                                segment[vid_self] = (x, y)
                            except:
                                pass
                            topology[j][vid_other] = (x, y)

        return topology

    def localise(self, x, y, heading, topology):
        """
        This function finds the next topology waypoint
        Along vehicle's heading
        """

        distance = self.distance_to_line(topology[0][0],
                                         topology[0][1], (x, y))
        nearest_segment = (distance, topology[0])
        for segment in topology:
            distance = self.distance_to_line(segment[0],
                                             segment[1], (x, y))
            if distance < nearest_segment[0]:
                nearest_segment = (distance, segment)
        segment = nearest_segment[1]

        heading_vector = (math.cos(heading), math.sin(heading))

        vector1 = self.unit_vector((x, y), segment[0])
        vector2 = self.unit_vector((x, y), segment[1])

        dot1 = self.dot(vector1, heading_vector)
        dot2 = self.dot(vector2, heading_vector)

        start_waypoint = None
        if dot1 > dot2:
            start_waypoint = segment[0]
        else:
            start_waypoint = segment[1]

        return start_waypoint

    def distance_to_line(self, point1, point2, target):
        """
        This functions finds the distance between the target point and 
        The line joining point1, point2
        """

        x1, y1 = point1
        x2, y2 = point2
        xt, yt = target
        m = (y2-y1)/(x2-x1)

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
