# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# This file contains the class for global route planning

"""
This module provides GlobalRoutePlanner implementation.
"""

import math
from heapq import heappop, heapify


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
        # self.topology = self.roundoff(self.topology)
        self.graph, self.id_map = self.build_graph(self.topology)

    def plan_route(self, origin, destination, heading):
        """
        The following function generates the route plan based on
        origin      : tuple containing x, y of the route's start position
        destination : tuple containing x, y of the route's end position
        heading     : current heading of the vehicle in radian

        return      : list of turn by turn navigation decision
        possible values are START, GO_STRAIGHT, LEFT, RIGHT,
        CHANGE_LANE_LEFT, CHANGE_LANE_RIGHT
        """

        xo, yo = origin
        xd, yd = destination

        start = self.localise(xo, yo,
                              self.topology, heading)[0]
        end1, end2 = self.localise(xd, yd,
                                   self.topology)

        route = self.graph_search(origin, start, end1, end2,
                                  self.graph, self.id_map)
        route = route[::-1]

        plan = []
        plan.append('START')
        cur_vector = self.unit_vector((xo, yo), route[0])
        for i in range(len(route)):
            if cur_vector is None:
                cur_vector = self.unit_vector(route[i-1], route[i])
            if i+1 < range(len(route)):
                next_vector = self.unit_vector(route[i], route[i+1])
            else:
                break
            angle = \
                math.atan2(*next_vector[::-1]) - math.atan2(*cur_vector[::-1])

        return None

    def graph_search(self, origin, start, end1, end2, graph, idmap):
        """
        This function perform's a Dijsktra's search from start to
        end nodes
        """
        q = []  # priority queue for Dijsktra's search
        visited = dict()    # visited node to through node map
        entry_lookup = dict()   # entry lookup for modifying q
        inf = float('inf')

        def d(a, b):
            return math.sqrt((a[0]-b[0])**2+(a[0]-b[0])**2)

        cnode = graph[idmap[start]]    # current node 
        for i in graph:
            node = graph[i]
            entry = [inf, [node.id, cnode.id]]
            entry_lookup[node.id] = entry
            q.append(entry)

        entry_lookup[cnode.id][0] = 0
        heapify(q)
        while idmap[end1] not in visited and idmap[end2] not in visited:
            popentry = heappop(q)
            popid = popentry[1][0]
            cnode = graph[popid]
            cd = popentry[0]    # current node distance from start
            via = popentry[1][1]    # through node
            visited[cnode.id] = [cd, via]
            for i in cnode.connections:
                node = graph[i]
                if via != cnode.id:
                    pre_vector = self.unit_vector(graph[via].vertex, 
                                                  cnode.vertex)
                else:
                    pre_vector = self.unit_vector(origin, cnode.vertex)
                new_vector = self.unit_vector(cnode.vertex, node.vertex)
                if self.dot(pre_vector, new_vector) >= 0:
                    entry = entry_lookup[node.id]
                    new_distance = d(node.vertex, cnode.vertex)+cd
                    if new_distance < entry[0]:
                        entry[0] = new_distance
                        entry[1][1] = cnode.id
                        heapify(q)

        endid1, endid2 = None, None
        if idmap[end1] in visited:
            endid1 = idmap[end1]
            endid2 = idmap[end2]
        else:
            endid1 = idmap[end2]
            endid2 = idmap[end1]

        route = []
        route.append(endid2)
        route.append(endid1)
        nextnode = visited[endid1][1]
        while nextnode != idmap[start]:
            route.append(nextnode)
            nextnode = visited[nextnode][1]
        route.append(idmap[start])

        return route

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
        list to 1cm. This is performed by comparing the co-ordinate pairs
        that are closer than 1cm to overcome edge case roundoff errors.
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

    def localise(self, x, y, topology, heading_vector=None):
        """
        This function finds the segment closest to (x, y)
        Optionally, it orders the segment with vertex
        along heading (in radians)

        TODO: Further test with visualizations
        """

        distance = float('inf')
        nearest_segment = (distance, None)
        for segment in topology:
            distance = self.distance_to_line(segment[0],
                                             segment[1], (x, y))
            # print(distance, segment)
            v1 = self.unit_vector((x, y), segment[0])
            v2 = self.unit_vector((x, y), segment[1])
            if self.dot(v1, v2) < 0 and distance < nearest_segment[0]:
                nearest_segment = (distance, segment)
        segment = nearest_segment[1]

        if heading_vector is not None and segment is not None:
            vector1 = self.unit_vector((x, y), segment[0])
            vector2 = self.unit_vector((x, y), segment[1])
            dot1 = self.dot(vector1, heading_vector)
            dot2 = self.dot(vector2, heading_vector)
            if dot1 > dot2:
                segment = (segment[0], segment[1])
            else:
                segment = (segment[1], segment[0])

        return segment

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
