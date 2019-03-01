#!/usr/bin/env python
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Object crash without prior vehicle action scenario:
The scenario realizes the user controlled ego vehicle
moving along the road and encountering a cyclist ahead.
"""

import py_trees

from srunner.scenariomanager.atomic_scenario_behavior import *
from srunner.scenariomanager.atomic_scenario_criteria import *
from srunner.scenariomanager.timer import TimeOut
from srunner.scenarios.basic_scenario import *


OBJECT_CROSSING_SCENARIOS = [
    "StationaryObjectCrossing",
    "DynamicObjectCrossing"
]


class StationaryObjectCrossing(BasicScenario):

    """
    This class holds everything required for a simple object crash
    without prior vehicle action involving a vehicle and a cyclist.
    The ego vehicle is passing through a road and encounters
    a stationary cyclist.
    """

    category = "ObjectCrossing"

    timeout = 30

    # ego vehicle parameters
    _ego_vehicle_velocity_allowed = 20
    _ego_vehicle_distance_driven = 35

    def __init__(self, world, ego_vehicle, other_actors, town, randomize=False, debug_mode=False, config=None):
        """
        Setup all relevant parameters and create scenario
        """
        super(StationaryObjectCrossing, self).__init__("Stationaryobjectcrossing",
                                                       ego_vehicle,
                                                       other_actors,
                                                       town,
                                                       world,
                                                       debug_mode)

    @staticmethod
    def initialize_actors(ego_vehicle):
        """
        This method returns the list of participant actors and their initial positions for the scenario
        """
        _start_distance = 40
        actor_parameters = []
        location, _ = get_location_in_distance(ego_vehicle, _start_distance)
        model = 'vehicle.diamondback.century'
        waypoint = ego_vehicle.get_world().get_map().get_waypoint(location)
        transform = carla.Transform(location, carla.Rotation(yaw=waypoint.transform.rotation.yaw+90))
        actor_parameters.append((model, transform))

        return actor_parameters

    def _create_behavior(self):
        """
        Only behavior here is to wait
        """
        redundant = TimeOut(self.timeout - 5)
        return redundant

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []

        max_velocity_criterion = MaxVelocityTest(self.ego_vehicle,
                                                 self._ego_vehicle_velocity_allowed,
                                                 optional=True)
        collision_criterion = CollisionTest(self.ego_vehicle)
        keep_lane_criterion = KeepLaneTest(self.ego_vehicle, optional=True)
        driven_distance_criterion = DrivenDistanceTest(self.ego_vehicle,
                                                       self._ego_vehicle_distance_driven,
                                                       distance_acceptable=30,
                                                       optional=True)

        criteria.append(max_velocity_criterion)
        criteria.append(collision_criterion)
        criteria.append(keep_lane_criterion)
        criteria.append(driven_distance_criterion)

        return criteria


class DynamicObjectCrossing(BasicScenario):

    """
    This class holds everything required for a simple object crash
    without prior vehicle action involving a vehicle and a cyclist,
    The ego vehicle is passing through a road,
    And encounters a cyclist crossing the road.
    """

    category = "ObjectCrossing"

    timeout = 60

    # ego vehicle parameters
    _ego_vehicle_velocity_allowed = 10
    _ego_vehicle_distance_driven = 50

    # other vehicle parameters
    _other_actor_target_velocity = 15
    _other_actor_max_brake = 1.0

    def __init__(self, world, ego_vehicle, other_actors, town, randomize=False, debug_mode=False):
        """
        Setup all relevant parameters and create scenario
        """

        super(DynamicObjectCrossing, self).__init__("Dynamicobjectcrossing",
                                                    ego_vehicle,
                                                    other_actors,
                                                    town,
                                                    world,
                                                    debug_mode)

    @staticmethod
    def initialize_actors(ego_vehicle):
        """
        This method returns the list of participant actors and their initial positions for the scenario
        """
        _start_distance = 40
        actor_parameters = []
        wmap = ego_vehicle.get_world().get_map()
        lane_width = wmap.get_waypoint(ego_vehicle.get_location()).lane_width
        location, _ = get_location_in_distance(ego_vehicle, _start_distance)
        model = 'vehicle.diamondback.century'

        offset = {
            "Town01" : {"orientation": 270, "position": 90, "z": 39, "coefficient": 0.6},
            "Town02" : {"orientation": 270, "position": 90, "z": 0, "coefficient": 2.2},
            "Town03" : {"orientation": -90, "position": 90, "z": 0, "coefficient": 0.1},
            "Town04" : {"orientation": 270, "position": 90, "z": 0, "coefficient": 0.1},
            "Town05" : {"orientation": 270, "position": 90, "z": 0, "coefficient": 1}
        }

        if wmap.name == "Town01": offset = offset["Town01"]
        elif wmap.name == "Town02": offset = offset["Town02"]
        elif wmap.name == "Town03": offset = offset["Town03"]
        elif wmap.name == "Town04": offset = offset["Town04"]
        elif wmap.name == "Town05": offset = offset["Town05"]
        else:
              print("No Town found")
        waypoint = wmap.get_waypoint(location)
        position_yaw = waypoint.transform.rotation.yaw+ offset['position']
        location.x += offset['coefficient']*lane_width*math.cos(position_yaw)
        location.y += offset['coefficient']*lane_width*math.sin(position_yaw)
        location.z += offset['z']
        transform = carla.Transform(location, carla.Rotation(yaw=offset['orientation']))
        actor_parameters.append((model, transform))

        return actor_parameters

    def _create_behavior(self):
        """
        After invoking this scenario, cyclist will wait for the user
        controlled vehicle to enter the in the trigger distance region,
        the cyclist starts crossing the road once the condition meets,
        then after 60 seconds, a timeout stops the scenario
        """
        # leaf nodes
        lane_width = self.ego_vehicle.get_world().get_map().get_waypoint(self.ego_vehicle.get_location()).lane_width
        start_condition = InTimeToArrivalToVehicle(self.other_actors[0], self.ego_vehicle, 13)
        keep_vel = KeepVelocity(self.other_actors[0], 3)
        keep_till = DriveDistance(self.other_actors[0], 0.3*lane_width)
        stop_other_actor = StopVehicle(self.other_actors[0], self._other_actor_max_brake)
        timeout_other = TimeOut(6)
        start_vehicle = AccelerateToVelocity(self.other_actors[0], 1.0,
                                             self._other_actor_target_velocity)
        trigger_other_actor = DriveDistance(self.other_actors[0], lane_width)
        stop_vehicle = StopVehicle(self.other_actors[0], self._other_actor_max_brake)
        timeout_other_actor = TimeOut(5)

         # non leaf nodes
        root = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        scenario_sequence = py_trees.composites.Sequence()
        keep_velocity_other = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        keep_velocity = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        # building tree
        root.add_child(scenario_sequence)
        scenario_sequence.add_child(HandBrakeVehicle(self.other_actors[0], True))
        scenario_sequence.add_child(start_condition)
        scenario_sequence.add_child(HandBrakeVehicle(self.other_actors[0], False))
        scenario_sequence.add_child(keep_velocity)
        scenario_sequence.add_child(stop_other_actor)
        scenario_sequence.add_child(timeout_other)
        scenario_sequence.add_child(keep_velocity_other)
        scenario_sequence.add_child(stop_vehicle)
        scenario_sequence.add_child(timeout_other_actor)
        keep_velocity.add_child(keep_vel)
        keep_velocity.add_child(keep_till)
        keep_velocity_other.add_child(start_vehicle)
        keep_velocity_other.add_child(trigger_other_actor)

        return root

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []

        max_velocity_criterion = MaxVelocityTest(self.ego_vehicle,
                                                 self._ego_vehicle_velocity_allowed,
                                                 optional=True)
        collision_criterion = CollisionTest(self.ego_vehicle)
        keep_lane_criterion = KeepLaneTest(self.ego_vehicle, optional=True)
        driven_distance_criterion = DrivenDistanceTest(self.ego_vehicle,
                                                       self._ego_vehicle_distance_driven,
                                                       distance_acceptable=30,
                                                       optional=True)

        criteria.append(max_velocity_criterion)
        criteria.append(collision_criterion)
        criteria.append(keep_lane_criterion)
        criteria.append(driven_distance_criterion)

        return criteria
