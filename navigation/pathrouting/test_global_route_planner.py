import math
import unittest
from mock import Mock
import carla
from global_route_planner import GlobalRoutePlanner
from global_route_planner_dao import GlobalRoutePlannerDAO


class Test_GlobalRoutePlanner(unittest.TestCase):
    """
    Test class for GlobalRoutePlanner class
    """

    def setUp(self):
        # == Utilities test instance without DAO == #
        self.simple_grp = GlobalRoutePlanner(None)

        # == Instance with sample DAO for graph testing == #
        input_topology = [[(1, 3), (1, 2)],
                          [(1, 2), (2, 2)],
                          [(2, 2), (2, 1)],
                          [(2, 1), (4, 1)],
                          [(4, 1), (4, 3)],
                          [(4, 3), (1, 3)],
                          [(4, 3), (1, 2)]]
        dao = GlobalRoutePlannerDAO(None)
        dao.get_topology = Mock(return_value=input_topology)
        dao.get_next_waypoint = Mock()
        wplookup = {(1.0, 2.5): (1.0, 2.0), (1.5, 2.0): (2.0, 2.0),
                    (2.0, 1.5): (2.0, 1.0), (3.0, 1.0): (4.0, 1.0),
                    (4.0, 2.0): (4.0, 3.0), (2.5, 3.0): (1.0, 3.0),
                    (2.5, 2.5): (1, 2.0)}
        dao.get_next_waypoint.side_effect = lambda *arg: wplookup[arg[0]]
        self.dao_grp = GlobalRoutePlanner(dao)
        self.dao_grp.setup()

        # == Integration test instance == #
        client = carla.Client('localhost', 2000)
        world = client.get_world()
        integ_dao = GlobalRoutePlannerDAO(world.get_map())
        self.integ_grp = GlobalRoutePlanner(integ_dao)
        self.integ_grp.setup()
        pass

    def tearDown(self):
        self.simple_grp = None
        self.dao_grp = None
        self.integ_grp = None
        pass

    def test_plan_route_town01(self):
        """
        Test for GlobalROutePlanner.plan_route()
        Run this test with a carla (0.9.1) instance running Town01
        """
        plan = self.integ_grp.plan_route((-60, -5), (-77.65, 72.72))
        self.assertEqual(
            plan, ['START', 'LEFT', 'LEFT', 'GO_STRAIGHT', 'LEFT', 'STOP'])

    def test_localise(self):
        """
        Test for GlobalROutePlanner.localise()
        """
        input_topology = [[(0, 1), (1, 0)],
                          [(2, 1), (3, 1)],
                          [(1, 1), (2, 1)],
                          [(2, 1), (1, 2)],
                          [(2, 2), (3, 2)]]
        x, y = (1.2, 1.01)
        nxt_vertex = self.simple_grp.localise(x, y, input_topology)[0]
        self.assertEqual(nxt_vertex, (1, 1))

    # def test_localise_town01(self):
    #     """
    #     Test for GlobalROutePlanner.localise()
    #     Run this test with a carla (0.9.1) instance running Town01
    #     """
    #     x, y = 334.7, 25
    #     segment = self.integ_grp.localise(x, y, self.integ_grp.topology)
    #     self.assertEqual(segment, [(334.6214904785156, 3.790151834487915),
    #                                (334.63958740234375, 53.424442291259766)])

    def test_build_graph(self):
        """
        Test for GlobalROutePlanner.build_graph()
        """
        graph, id_map = self.dao_grp.build_graph()

        def connection_check(n1, n2):
            return graph.has_edge(n1, n2)

        self.assertEqual(len(id_map), 6)
        self.assertTrue(connection_check(0, 1))
        self.assertTrue(connection_check(1, 2))
        self.assertTrue(connection_check(2, 3))
        self.assertTrue(connection_check(3, 4))
        self.assertTrue(connection_check(4, 5))
        self.assertTrue(connection_check(5, 0))
        self.assertTrue(connection_check(5, 1))

    def test_distance_to_line(self):
        """
        Test for GlobalROutePlanner.distance_to_line()
        """
        dist = self.simple_grp.distance_to_line((0, 0), (2, 2), (2, 0))
        self.assertEqual(round(dist, 3), round(math.sqrt(2), 3))

    def test_unit_vector(self):
        """
        Test for GlobalROutePlanner.unit_vector()
        """
        vector = self.simple_grp.unit_vector((1, 1), (2, 2))
        self.assertAlmostEquals(vector[0], 1 / math.sqrt(2))
        self.assertAlmostEquals(vector[1], 1 / math.sqrt(2))

    def test_dot(self):
        """
        Test for GlobalROutePlanner.test_dot()
        """
        self.assertAlmostEqual(self.simple_grp.dot((1, 0), (0, 1)), 0)
        self.assertAlmostEqual(self.simple_grp.dot((1, 0), (1, 0)), 1)


def suite():
    """
    Gathering all tests
    """

    suite = unittest.TestSuite()
    suite.addTest(Test_GlobalRoutePlanner('test_unit_vector'))
    suite.addTest(Test_GlobalRoutePlanner('test_dot'))
    suite.addTest(Test_GlobalRoutePlanner('test_distance_to_line'))
    suite.addTest(Test_GlobalRoutePlanner('test_build_graph'))
    suite.addTest(Test_GlobalRoutePlanner('test_localise'))
    suite.addTest(Test_GlobalRoutePlanner('test_path_search'))
    suite.addTest(Test_GlobalRoutePlanner('test_plan_route'))

    return suite


if __name__ == '__main__':
    """
    Running test suite
    """
    mySuit = suite()
    runner = unittest.TextTestRunner()
    runner.run(mySuit)
