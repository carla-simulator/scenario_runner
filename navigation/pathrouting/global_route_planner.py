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
        # Transforming topology into list of vertex pairs
        for segment in self.world.get_map().get_topology():
            x1 = segment[0].transform.location.x
            y1 = segment[0].transform.location.y
            x2 = segment[1].transform.location.x
            y2 = segment[1].transform.location.y
            self.topology.append([(x1, y1), (x2, y2)])
        pass
        # Creating graph of the world map and also a map from 
        # node co-ordinates to node id
        self.graph, self.id_map = self.build_graph(self.topology)

    def plan_route(self, origin, heading, destination, graph, idmap, topology):
        """
        The following function generates the route plan based on
        origin      : tuple containing x, y of the route's start position
        destination : tuple containing x, y of the route's end position
        heading     : current heading of the vehicle in radian

        return      : list of turn by turn navigation decision
        possible values (for now) are START, GO_STRAIGHT, LEFT, RIGHT,
        STOP
        """

        xo, yo = origin
        xd, yd = destination
        start = self.localise(xo, yo, topology, heading)
        end = self.localise(xd, yd, topology)
        route = self.graph_search(start, end, graph, idmap)
        route = route[::-1]

        plan = []
        plan.append('START')
        for i in [x for x in range(len(route)-2) if x%2 == 0]:
            v1 = self.unit_vector(graph[route[i]].vertex,
                                  graph[route[i+1]].vertex)
            v2 = self.unit_vector(graph[route[i+1]].vertex,
                                  graph[route[i+2]].vertex)
            direction = math.atan2(*v2[::-1]) - math.atan2(*v1[::-1])
            if abs(direction) < 0.174533:
                plan.append('GO_STRAIGHT')
            elif direction > 0:
                plan.append('LEFT')
            elif direction < 0:
                plan.append('RIGHT')
        plan.append('STOP')

        return plan

    def graph_search(self, start, end, graph, idmap):
        """
        This function perform's a Dijsktra's search from start to end nodes.
        start   :   the road segment containing origin
        end     :   the orad segment containing destination

        return  :   list of nodes connecting start and end
        """
        
        start1, start2 = start
        end1, end2 = end
        q = []  # priority queue for Dijsktra's search
        visited = dict()    # visited node to through node map
        entry_lookup = dict()   # map from node id to queue entry
        inf = float('inf')

        def d(a, b):
            return math.sqrt((a[0]-b[0])**2+(a[0]-b[0])**2)

        # Initializing priority queue for Dijsktra's search
        cnode = graph[idmap[start1]]    # current node 
        for i in graph:
            node = graph[i]
            entry = [inf, [node.id, cnode.id]]
            entry_lookup[node.id] = entry
            q.append(entry)
        entry_lookup[cnode.id][0] = 0
        heapify(q)

        # Performing Dijsktra's search
        while idmap[end1] not in visited and idmap[end2] not in visited:
            popentry = heappop(q)
            popid = popentry[1][0]
            cnode = graph[popid]
            cd = popentry[0]    # current node distance from start
            via = popentry[1][1]    # through node id
            visited[cnode.id] = [cd, via]
            for i in cnode.connections:
                node = graph[i]
                if via != cnode.id:
                    pre_vector = self.unit_vector(graph[via].vertex, 
                                                  cnode.vertex)
                else:
                    pre_vector = self.unit_vector(start2, cnode.vertex)
                new_vector = self.unit_vector(cnode.vertex, node.vertex)
                if self.dot(pre_vector, new_vector) >= 0:
                    entry = entry_lookup[node.id]
                    new_distance = d(node.vertex, cnode.vertex)+cd
                    if new_distance < entry[0]:
                        entry[0] = new_distance
                        entry[1][1] = cnode.id
                        heapify(q)

        # Checking which vertex of the end segment was reached first
        # for appending the other vertex to route in proper order
        endid1, endid2 = None, None
        if idmap[end1] in visited:
            endid1 = idmap[end1]
            endid2 = idmap[end2]
        else:
            endid1 = idmap[end2]
            endid2 = idmap[end1]

        # Building route
        route = []
        route.append(endid2)
        route.append(endid1)
        nextnode = visited[endid1][1]
        while nextnode != idmap[start1]:
            route.append(nextnode)
            nextnode = visited[nextnode][1]
        route.append(idmap[start1])
        route.append(idmap[start2])

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
            self.vertex = vertex  # vertex co-ordinates as a tuple
            self.connections = []   # list of connecting node ids

        def add_connection(self, connecting_node_id):
            self.connections.append(connecting_node_id)
            pass
        pass

    def localise(self, x, y, topology, heading_vector=None):
        """
        This function finds the road segment closest to (x, y)
        Optionally, it orders the segment with vertex
        along heading_vector at position 0
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

        # Ordering the segment vertices along heading_vector
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
