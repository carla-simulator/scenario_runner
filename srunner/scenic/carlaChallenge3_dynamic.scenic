""" Scenario Description
Traffic Scenario 03 (dynamic).
Obstacle avoidance without prior action.
The ego-vehicle encounters an obstacle / unexpected entity on the road and must perform an
emergency brake or an avoidance maneuver.
"""

# SET MAP AND MODEL (i.e. definitions of all referenceable vehicle types, road library, etc)
param map = localPath('assets/Town10HD_Opt.xodr')
param carla_map = 'Town10HD_Opt'
model srunner.scenic.models.model

## CONSTANTS
EGO_MODEL = "vehicle.lincoln.mkz"
EGO_SPEED = 7
SAFETY_DISTANCE = 17
BRAKE_INTENSITY = 1.0

PEDESTRIAN_MIN_SPEED = 15
THRESHOLD = 30

# EGO BEHAVIOR: Follow lane and brake when reaches threshold distance to obstacle
behavior EgoBehavior(speed=10):
    try:
        do FollowLaneBehavior(target_speed=speed)
    interrupt when withinDistanceToObjsInLane(self, SAFETY_DISTANCE):
        take SetBrakeAction(BRAKE_INTENSITY)

behavior PedestrianBehavior(min_speed=1, threshold=10):
    do CrossingBehavior(ego, min_speed, threshold)

## DEFINING SPATIAL RELATIONS
# Please refer to scenic/domains/driving/roads.py how to access detailed road infrastructure
# 'network' is the 'class Network' object in roads.py 

# make sure to put '*' to uniformly randomly select from all elements of the list, 'network.lanes'
lane = Uniform(*network.lanes)

spot = new OrientedPoint on lane.centerline
vending_spot = new OrientedPoint following roadDirection from spot for -3

pedestrian = new Pedestrian right of spot by 3,
    with heading 90 deg relative to spot.heading,
    with regionContainedIn None,
    with behavior PedestrianBehavior(PEDESTRIAN_MIN_SPEED, THRESHOLD)

vending_machine = new VendingMachine right of vending_spot by 3,
    with heading -90 deg relative to vending_spot.heading,
    with regionContainedIn None

ego = new Car following roadDirection from spot for Range(-40, -35),
    with blueprint EGO_MODEL,
    with behavior EgoBehavior(EGO_SPEED)

require (distance to intersection) > 50
require (ego.laneSection._slowerLane is None)
terminate when (distance to spot) > 40