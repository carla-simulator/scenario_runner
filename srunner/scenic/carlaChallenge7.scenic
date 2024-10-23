""" Scenario Description
Based on 2019 Carla Challenge Traffic Scenario 07.
Ego-vehicle is going straight at an intersection but a crossing vehicle 
runs a red light, forcing the ego-vehicle to perform a collision avoidance maneuver.
Note: The traffic light control is not implemented yet, but it will soon be. 
"""
param map = localPath('assets/Town10HD_Opt.xodr')
param carla_map = 'Town10HD_Opt'
model srunner.scenic.models.model

EGO_MODEL = "vehicle.lincoln.mkz"

SAFETY_DISTANCE = 15
BRAKE_INTENSITY = 1.0
DISTANCE_TO_INTERSECTION = Uniform(10, 15) * -1

behavior CrossingCarBehavior(trajectory):
    do FollowTrajectoryBehavior(trajectory=trajectory)
    wait

behavior EgoBehavior(trajectory):

    try:
        do FollowTrajectoryBehavior(target_speed=7, trajectory=trajectory)
        terminate
    interrupt when withinDistanceToAnyObjs(self, SAFETY_DISTANCE):
        take SetBrakeAction(BRAKE_INTENSITY)

fourWayIntersection = filter(lambda i: i.is4Way, network.intersections)
intersec = Uniform(*fourWayIntersection)

# Get the ego manuever
startLane = Uniform(*intersec.incomingLanes)
ego_maneuvers = filter(lambda i: i.type == ManeuverType.STRAIGHT, startLane.maneuvers)
ego_maneuver = Uniform(*ego_maneuvers)
ego_trajectory = [ego_maneuver.startLane, ego_maneuver.connectingLane, ego_maneuver.endLane]

# Get the adversary maneuver
other_maneuvers = filter(lambda i: i.type == ManeuverType.STRAIGHT, ego_maneuver.conflictingManeuvers)
other_maneuver = Uniform(*other_maneuvers)
other_trajectory = [other_maneuver.startLane, other_maneuver.connectingLane, other_maneuver.endLane]

## OBJECT PLACEMENT
other_spwPt = new OrientedPoint in other_maneuver.startLane.centerline
ego_spwPt = new OrientedPoint in ego_maneuver.startLane.centerline

ego = new Car at ego_spwPt,
    with blueprint EGO_MODEL,
    with behavior EgoBehavior(trajectory=ego_trajectory)

crossing_car = new Truck at other_spwPt,
    with blueprint 'vehicle.ambulance.ford',
    with behavior CrossingCarBehavior(other_trajectory)

require 10 <= (distance to intersec) <= 15
terminate when (distance to ego_spwPt) >= 40