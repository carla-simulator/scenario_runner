""" Scenario Description
Traffic Scenario 10.
Crossing negotiation at an unsignalized intersection.
The ego-vehicle needs to negotiate with other vehicles to cross an unsignalized intersection. In
this situation it is assumed that the first to enter the intersection has priority.
"""

## SET MAP AND MODEL (i.e. definitions of all referenceable vehicle types, road library, etc)
param map = localPath('assets/Town10HD_Opt.xodr')
param carla_map = 'Town10HD_Opt'
model srunner.scenic.models.model

## CONSTANTS
EGO_MODEL = "vehicle.lincoln.mkz"
EGO_SPEED = 10
SAFETY_DISTANCE = 20
BRAKE_INTENSITY = 1.0

##DEFINING BEHAVIORS
behavior AdversaryBehavior(trajectory):
    do FollowTrajectoryBehavior(trajectory=trajectory)

behavior EgoBehavior(speed, trajectory):
    try:
        do FollowTrajectoryBehavior(target_speed=speed, trajectory=trajectory)
        do FollowLaneBehavior(target_speed=speed)
    interrupt when withinDistanceToAnyObjs(self, SAFETY_DISTANCE):
        take SetBrakeAction(BRAKE_INTENSITY)

## DEFINING SPATIAL RELATIONS
# Please refer to scenic/domains/driving/roads.py how to access detailed road infrastructure
# 'network' is the 'class Network' object in roads.py

fourWayIntersection = filter(lambda i: i.is4Way and not i.isSignalized, network.intersections)

# make sure to put '*' to uniformly randomly select from all elements of the list
intersec = Uniform(*fourWayIntersection)
ego_start_lane = Uniform(*intersec.incomingLanes)

ego_maneuver = Uniform(*ego_start_lane.maneuvers)
ego_trajectory = [ego_maneuver.startLane, ego_maneuver.connectingLane, ego_maneuver.endLane]

adv_maneuver = Uniform(*ego_maneuver.conflictingManeuvers)
adv_trajectory = [adv_maneuver.startLane, adv_maneuver.connectingLane, adv_maneuver.endLane]
adv_start_lane = adv_maneuver.startLane

## OBJECT PLACEMENT
ego_spawn_pt = new OrientedPoint in ego_maneuver.startLane.centerline
adv_spawn_pt = new OrientedPoint in adv_maneuver.startLane.centerline

ego = new Car at ego_spawn_pt,
    with blueprint EGO_MODEL,
    with behavior EgoBehavior(EGO_SPEED, ego_trajectory)

adversary = new Car at adv_spawn_pt,
    with behavior AdversaryBehavior(adv_trajectory)

require 20 <= (distance to intersec) <= 25
require 15 <= (distance from adversary to intersec) <= 20
terminate when (distance to ego_spawn_pt) > 70
