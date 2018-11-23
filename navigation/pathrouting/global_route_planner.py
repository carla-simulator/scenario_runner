# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# This file contains class for global route planning

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
        Process the topology returned by world into a list of simple co-ordinate pairs
        """
        self.world = world
        self.topology = []
        world_map = self.world.get_map()
        topology = world_map.get_topology()
        for segment in topology:
            x1, y1 = segment[0].transform.location.x, segment[0].transform.location.y
            x2, y2 = segment[1].transform.location.x, segment[1].transform.location.y
            self.topology.append(((x1,y1),(x2,y2)))
        pass
        self.graph = self.__build_graph__(self.topology)

    def plan_route(self, origin, destination, heading):
        """
        The following function generates the route plan based on
        origin      : tuple containing x, y co-ordinates of the route's start position
        destination : tuple containing x, y co-ordinates of the route's end position
        heading     : current heading of the vehicle in radian

        return      : list of turn by turn navigation decision 
        possible values are GO_STRAIGHT,LEFT,RIGHT,CHANGE_LANE_LEFT,CHANGE_LANE_RIGHT
        """

        x_origin, y_origin = origin
        x_destination, y_destination = destination

        x_start, y_start = self.__find_start_waypoint__(x_origin, y_origin, heading)

        return None
    
    def __build_graph__(self, topology):
        
        # Structure of graph dictionary:
        # {node_id: node object}
        graph = dict()
        distance_check = lambda p1, p2: math.sqrt((p2[0]-p1[0])**2+(p2[1]-p1[1])**2) < 1
        for i, segment in enumerate(topology):
            graph[i] = self.__node__(i, segment)
            
            for j in range(i+1, len(topology)):
                
                for vertex_id_self, vertex_self in enumerate(segment):
                    for vertex_id_other, vertex_other in enumerate(topology[j]):
                        
                        if distance_check(vertex_self, vertex_other):
                            connection_found = True
                            connecting_node_id = int(str(i)+str(j))
                            graph[i].add_connection(connecting_node_id, vertex_id_self, vertex_id_other)
                            if connecting_node_id not in graph:
                                graph[connecting_node_id] = self.__node__(connecting_node_id, topology[j])
                            graph[connecting_node_id].add_connection(i, vertex_id_other, vertex_id_self)
                            
                            break
                    if connection_found:
                        break
                pass
            pass

        return graph

    class __node__(object):
        """
        node object in the topology graph
        """
        
        def __init__(self, id, segment):
            self.id = id
            # vertex dictionary maps vertex id to tupple containing co-ordinates
            self.vertex = {0: segment[0], 1: segment[1]}
            # structure of connections dictionary
            # {vertex1_id : {node_id: vertex_id of connecting node}, vertex2_id: { ... }}
            self.connections = {0: dict(), 1: dict()}

        def add_connection(self, connecting_node_id, vertex_id_self, vertex_id_other):
            self.connections[vertex_id_self][connecting_node_id] = vertex_id_other
            pass
        pass

    def __find_start_waypoint__(self, x, y, heading):
        """
        This function finds the next topology waypoint the vehicle should move towards
        """
        
        distance = self.__distance_from_segment__(self.topology[0][0],self.topology[0][1], (x,y))
        nearest_segment = (distance, self.topology[0])
        for segment in self.topology:
            distance = self.__distance_from_segment__(segment[0], segment[1], (x,y))
            if distance < nearest_segment[0]:
                nearest_segment = (distance, segment)
        segment = nearest_segment[1]

        heading_vector = (math.cos(heading), math.sin(heading))

        vector1 = self.__unit_vector__((x,y), segment[0])
        vector2 = self.__unit_vector__((x,y), segment[1])

        dot1 = self.__dot__(vector1, heading)
        dot2 = self.__dot__(vector2, heading_vector)

        start_waypoint = None
        if dot1 > dot2:
            start_waypoint = segment[0]
        else:
            start_waypoint = segment[1]

        return start_waypoint

    def __distance_from_segment__(self, point1, point2, target):
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

    def __unit_vector__(self, point1, point2):
        """
        This function returns the unit vector from point1 to point2
        """
        
        x1, y1 = point1
        x2, y2 = point2

        vector = (x2-x1, y2-y1)
        vector_mag = math.sqrt(vector[0]**2+vector[1]**2)
        vector = (vector[0]/vector_mag, vector[1]/vector_mag)

        return vector

    def __dot__(self, vector1, vector2):
        """
        This function returns the dot product of vector1 with vector2
        """
        return vector1[0]*vector2[0]+vector1[1]*vector2[1]

    pass
