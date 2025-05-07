""" Scenario Description
Based on 2019 Carla Challenge Traffic Scenario 09.
Ego-vehicle is performing a right turn at an intersection, yielding to crossing traffic.
"""
param map = localPath('assets/Town10HD_Opt.xodr')
param carla_map = 'Town10HD_Opt'
model srunner.scenic.models.model

EGO_MODEL = "vehicle.lincoln.mkz"

DISTANCE_TO_INTERSECTION1 = Uniform(5, 10) * -1
DISTANCE_TO_INTERSECTION2 = Uniform(35, 40) * -1
SAFETY_DISTANCE = 20
BRAKE_INTENSITY = 1.0


behavior CrossingCarBehavior(trajectory):
    do FollowTrajectoryBehavior(trajectory=trajectory)
    wait

behavior EgoBehavior(trajectory):
    try :
        do FollowTrajectoryBehavior(target_speed=7, trajectory=trajectory)
    interrupt when withinDistanceToAnyObjs(self, SAFETY_DISTANCE):
        take SetBrakeAction(BRAKE_INTENSITY)

spawnAreas = []
fourWayIntersection = filter(lambda i: i.is4Way, network.intersections)
intersec = Uniform(*fourWayIntersection)

startLane = Uniform(*intersec.incomingLanes)
ego_maneuvers = filter(lambda i: i.type == ManeuverType.RIGHT_TURN, startLane.maneuvers)
ego_maneuver = Uniform(*ego_maneuvers)
ego_trajectory = [ego_maneuver.startLane, ego_maneuver.connectingLane, ego_maneuver.endLane]

other_maneuvers = filter(lambda i: i.type == ManeuverType.STRAIGHT, ego_maneuver.conflictingManeuvers)
other_maneuver = Uniform(*other_maneuvers)
other_trajectory = [other_maneuver.startLane, other_maneuver.connectingLane, other_maneuver.endLane]

## OBJECT PLACEMENT
ego_spawn_pt = new OrientedPoint in ego_maneuver.startLane.centerline
other_spawn_pt = new OrientedPoint in other_maneuver.startLane.centerline

crossing_car = new Car at other_spawn_pt,
    with behavior CrossingCarBehavior(other_trajectory)

ego = new Car at ego_spawn_pt,
    with blueprint EGO_MODEL,
    with behavior EgoBehavior(ego_trajectory)

require 35 <= (distance to intersec) <= 40
# require 10 <= (distance from crossing_car to intersec) <= 15
terminate when (distance to ego_spawn_pt) >= 70