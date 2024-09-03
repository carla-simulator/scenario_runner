""" Scenario Description
Based on 2019 Carla Challenge Traffic Scenario 05.
Ego-vehicle performs a lane changing to evade a leading vehicle, which is moving too slowly.
"""
param map = localPath('assets/Town10HD_Opt.xodr')
param carla_map = 'Town10HD_Opt'
model srunner.scenic.models.model

#CONSTANTS
EGO_SPEED = 10
SLOW_CAR_SPEED = 6
EGO_TO_BICYCLE = 10
DIST_THRESHOLD = 15

#EGO BEHAVIOR: Follow lane, then perform a lane change
behavior EgoBehavior(leftpath, origpath=[]):
    laneChangeCompleted = False

    try: 
        do FollowLaneBehavior(EGO_SPEED)

    interrupt when withinDistanceToAnyObjs(self, DIST_THRESHOLD) and not laneChangeCompleted:
        do LaneChangeBehavior(laneSectionToSwitch=leftpath, target_speed=10)
        laneChangeCompleted = True

#OTHER BEHAVIOR
behavior SlowCarBehavior():
    do FollowLaneBehavior(SLOW_CAR_SPEED)

#GEOMETRY
laneSecsWithRightLane = []
for lane in network.lanes:
    for laneSec in lane.sections:
        if laneSec._laneToRight != None:
            laneSecsWithRightLane.append(laneSec)

assert len(laneSecsWithRightLane) > 0, \
    'No lane sections with adjacent left lane in network.'

initLaneSec = Uniform(*laneSecsWithRightLane)
rightLane = initLaneSec._laneToRight

#PLACEMENT
spawnPt = new OrientedPoint on initLaneSec.centerline

ego = new Car at spawnPt,
    with behavior EgoBehavior(rightLane, [initLaneSec])

cyclist = new Car following roadDirection from ego for EGO_TO_BICYCLE,
    with behavior SlowCarBehavior()

require (distance from ego to intersection) > 10
require (distance from cyclist to intersection) > 10