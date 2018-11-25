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

    def test_build_graph(self):
        input_topology = [[(1, 3), (1, 2)],
                          [(1, 2), (2, 2)],
                          [(2, 2), (2, 1)],
                          [(2, 1), (4, 1)],
                          [(4, 1), (4, 3)],
                          [(4, 3), (1, 3)]]
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

    def test_distance_to_line(self):
        dist = self.grp.distance_to_line((0, 0), (2, 2), (2, 0))
        self.assertAlmostEquals(dist, math.sqrt(2))

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
    # suite.addTest(Test_GlobalRoutePlanner('test___find_start_waypoint__'))

    return suite

if __name__ == '__main__':

    mySuit = suite()
    runner = unittest.TextTestRunner()
    runner.run(mySuit)
