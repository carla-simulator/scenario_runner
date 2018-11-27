import unittest
from global_route_planner import GlobalRoutePlanner
import carla
import math


class Test_GlobalRoutePlanner(unittest.TestCase):
    """
    Test class for GlobalRoutePlanner class
    """

    def setUp(self):
        client = carla.Client('localhost', 2000)
        world = client.get_world()
        self.grp = GlobalRoutePlanner(world)

    def tearDown(self):
        self.grp = None

    def test_graph_search(self):
        input_topology = [[(1, 3), (1, 2)],
                          [(1, 2), (2, 2)],
                          [(2, 2), (2, 1)],
                          [(2, 1), (4, 1)],
                          [(4, 1), (4, 3)],
                          [(4, 3), (1, 3)],
                          [(4, 3), (1, 2)]]
        xo, yo = 1, 2.9
        xd, yd = 3, 0.9
        heading = (0, -1)
        start = self.grp.localise(xo, yo, input_topology, heading)
        end = self.grp.localise(xd, yd, input_topology)
        graph, id_map = self.grp.build_graph(input_topology)
        route = self.grp.graph_search(start, end, graph, id_map)
        self.assertEqual(route, [4, 3, 2, 1, 0])
        
    def test_graph_search_town01(self):
        xo, yo = 120, -2.27
        xd, yd = 334.7, 165
        heading = (1, 0)
        start = self.grp.localise(xo, yo, self.grp.topology, heading)
        end = self.grp.localise(xd, yd, self.grp.topology)
        graph, idmap = self.grp.build_graph(self.grp.topology)
        route = self.grp.graph_search(start, end, graph, idmap)
        self.assertEqual(route,
                         [10, 12, 13, 49, 80, 81, 84,
                          85, 69, 70, 73, 74, 63, 65])

    def test_localise(self):
        input_topology = [[(0, 1), (1, 0)],
                          [(2, 1), (3, 1)],
                          [(1, 1), (2, 1)],
                          [(2, 1), (1, 2)],
                          [(2, 2), (3, 2)]]
        x, y = (1.2, 1.01)
        heading = (-1, 0)
        xn, yn = self.grp.localise(x, y, input_topology, heading)[0]
        self.assertEqual(xn, 1)
        self.assertEqual(yn, 1)
        # testing on Town01 map
        x, y = 334.7, 25
        segment = self.grp.localise(x, y, self.grp.topology)
        self.assertEqual(segment, [(334.6214904785156, 3.790151834487915),
                                   (334.63958740234375, 53.424442291259766)])

    def test_build_graph(self):
        input_topology = [[(1, 3), (1, 2)],
                          [(1, 2), (2, 2)],
                          [(2, 2), (2, 1)],
                          [(2, 1), (4, 1)],
                          [(4, 1), (4, 3)],
                          [(4, 3), (1, 3)],
                          [(4, 3), (1, 2)]]
        graph, id_map = self.grp.build_graph(input_topology)

        def connection_check(n1, n2):
            return n2 in graph[n1].connections and n1 in graph[n2].connections
        self.assertEqual(len(id_map), 6)
        self.assertTrue(connection_check(0, 1))
        self.assertTrue(connection_check(1, 2))
        self.assertTrue(connection_check(2, 3))
        self.assertTrue(connection_check(3, 4))
        self.assertTrue(connection_check(4, 5))
        self.assertTrue(connection_check(5, 0))
        self.assertTrue(connection_check(5, 1))

    def test_distance_to_line(self):
        dist = self.grp.distance_to_line((0, 0), (2, 2), (2, 0))
        self.assertEqual(round(dist, 3), round(math.sqrt(2), 3))

    def test_unit_vector(self):
        vector = self.grp.unit_vector((1, 1), (2, 2))
        self.assertAlmostEquals(vector[0], 1/math.sqrt(2))
        self.assertAlmostEquals(vector[1], 1/math.sqrt(2))

    def test_dot(self):
        self.assertAlmostEqual(self.grp.dot((1, 0), (0, 1)), 0)
        self.assertAlmostEqual(self.grp.dot((1, 0), (1, 0)), 1)


def suite():
    """
    Gathering all test cases
    """

    suite = unittest.TestSuite()
    suite.addTest(Test_GlobalRoutePlanner('test_unit_vector'))
    suite.addTest(Test_GlobalRoutePlanner('test_dot'))
    suite.addTest(Test_GlobalRoutePlanner('test_distance_to_line'))
    suite.addTest(Test_GlobalRoutePlanner('test_build_graph'))
    suite.addTest(Test_GlobalRoutePlanner('test_localise'))

    return suite

if __name__ == '__main__':

    mySuit = suite()
    runner = unittest.TextTestRunner()
    runner.run(mySuit)
