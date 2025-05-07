""" Scenario Description
Traffic Scenario 02.
Longitudinal control after leading vehicle’s brake.
The leading vehicle decelerates suddenly due to an obstacle and the ego-vehicle must perform an
emergency brake or an avoidance maneuver.
"""

## SET MAP AND MODEL (i.e. definitions of all referenceable vehicle types, road library, etc)
param map = localPath('assets/Town10HD_Opt.xodr')
param carla_map = 'Town10HD_Opt'
model srunner.scenic.models.model

## CONSTANTS
EGO_MODEL = "vehicle.lincoln.mkz"
EGO_SPEED = 7
EGO_BRAKING_THRESHOLD = 12

LEAD_CAR_SPEED = 8
LEADCAR_BRAKING_THRESHOLD = 15

BRAKE_ACTION = 1.0

## DEFINING BEHAVIORS
# EGO BEHAVIOR: Follow lane, and brake after passing a threshold distance to the leading car
behavior EgoBehavior(speed=10):
    try:
        do FollowLaneBehavior(speed)

    interrupt when withinDistanceToAnyCars(self, EGO_BRAKING_THRESHOLD):
        take SetBrakeAction(BRAKE_ACTION)

# LEAD CAR BEHAVIOR: Follow lane, and brake after passing a threshold distance to obstacle
behavior LeadingCarBehavior(speed=10):
    try: 
        do FollowLaneBehavior(speed)

    interrupt when withinDistanceToAnyObjs(self, LEADCAR_BRAKING_THRESHOLD):
        take SetBrakeAction(BRAKE_ACTION)

## DEFINING SPATIAL RELATIONS
# Please refer to scenic/domains/driving/roads.py how to access detailed road infrastructure
# 'network' is the 'class Network' object in roads.py

# make sure to put '*' to uniformly randomly select from all elements of the list, 'lanes'
lane = Uniform(*network.lanes)

obstacle = new Trash on lane.centerline

leadCar = new Car following roadDirection from obstacle for Range(-60, -50),
        with behavior LeadingCarBehavior(LEAD_CAR_SPEED)

ego = new Car following roadDirection from leadCar for Range(-15, -10),
        with blueprint EGO_MODEL,
        with behavior EgoBehavior(EGO_SPEED)

require (distance to intersection) > 60
terminate when ego.speed < 0.1 and (distance to obstacle) < 30
