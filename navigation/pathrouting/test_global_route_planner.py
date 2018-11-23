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

    def test___distance_from_segment__(self):
        self.assertAlmostEquals(self.grp.__distance_from_segment__((0,0), (2,2), (2,0)), math.sqrt(2))
    
    def test___unit_vector__(self):
        vector = self.grp.__unit_vector__((1,1), (2,2))
        self.assertAlmostEquals(vector[0], 1/math.sqrt(2))
        self.assertAlmostEquals(vector[1], 1/math.sqrt(2))

    def test___dot__(self):
        self.assertAlmostEqual(self.grp.__dot__((1,0), (0,1)), 0)
        self.assertAlmostEqual(self.grp.__dot__((1,0), (1,0)), 1)

    # def test___find_start_waypoint__(self):
    #     self.assertEquals( 1, 1)

def suite():
    """
    Gathering all test cases
    """
    
    suite = unittest.TestSuite()
    suite.addTest(Test_GlobalRoutePlanner('test___distance_from_segment__'))
    suite.addTest(Test_GlobalRoutePlanner('test___unit_vector__'))
    suite.addTest(Test_GlobalRoutePlanner('test___dot__'))
    # suite.addTest(Test_GlobalRoutePlanner('test___find_start_waypoint__'))
    
    return suite

if __name__ == '__main__':
    
    mySuit=suite()
    runner=unittest.TextTestRunner()
    runner.run(mySuit)