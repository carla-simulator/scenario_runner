""" Scenario Description
Cut-in from the right lane.
A vehicle driving ahead of the ego in the right lane changes into the ego's
lane, and the ego must brake to keep its distance. Where it happens on the map,
how far ahead the other vehicle starts, both speeds and how long it waits
before changing lane are sampled, so some samples have the ego already level
with it and the vehicle cuts in behind.
"""

## SET MAP AND MODEL (i.e. definitions of all referenceable vehicle types, road library, etc)
param map = localPath('assets/Town10HD_Opt.xodr')
param carla_map = 'Town10HD_Opt'
param timestep = 0.05
model srunner.scenic.models.model

## CONSTANTS
EGO_MODEL = "vehicle.lincoln.mkz"
EGO_SPEED = Range(8, 11)
EGO_BRAKING_THRESHOLD = 9

CUTIN_MODEL = "vehicle.tesla.model3"
CUTIN_SPEED = Range(6, 9)
CUTIN_DELAY = Range(0.5, 3.0)

## DEFINING BEHAVIORS
behavior EgoBehavior():
    try:
        do FollowLaneBehavior(target_speed=EGO_SPEED)
    interrupt when withinDistanceToObjsInLane(self, EGO_BRAKING_THRESHOLD):
        take SetBrakeAction(0.8)

behavior CutInBehavior(speed, delay):
    do FollowLaneBehavior(target_speed=speed) for delay seconds
    # The car has moved on since it spawned, so read the lane from where it is now.
    do LaneChangeBehavior(laneSectionToSwitch=self.laneSection._laneToLeft, target_speed=speed)
    do FollowLaneBehavior(target_speed=speed)

## DEFINING SPATIAL RELATIONS
# Only lane sections know their neighbours. Keep those whose right neighbour
# runs the same way as they do.
laneSecs = []
for lane in network.lanes:
    for sec in lane.sections:
        if sec._laneToRight != None and sec._laneToRight.isForward == sec.isForward:
            laneSecs.append(sec)
assert len(laneSecs) > 0, 'no lane section with a same-direction right neighbour'
initLaneSec = Uniform(*laneSecs)
adjacentLane = initLaneSec._laneToRight
ego_spawn = new OrientedPoint on initLaneSec.centerline

## SCENARIO SPECIFICATION
ego = new Car at ego_spawn,
    with blueprint EGO_MODEL,
    with behavior EgoBehavior()

cutin_car = new Car on adjacentLane.centerline,
    with blueprint CUTIN_MODEL,
    with behavior CutInBehavior(CUTIN_SPEED, CUTIN_DELAY)

# More than a car length ahead, or the two overlap side by side and the lane
# change steers into the ego.
require (relative position of cutin_car from ego).y > 8
require (distance from ego to cutin_car) < 20
require (distance to intersection) > 40
require (distance from cutin_car to intersection) > 30
terminate when (distance to ego_spawn) > 70
