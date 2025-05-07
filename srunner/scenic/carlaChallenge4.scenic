param map = localPath('assets/Town10HD_Opt.xodr')
param carla_map = 'Town10HD_Opt'
model srunner.scenic.models.model

EGO_MODEL = "vehicle.lincoln.mkz"
PEDESTRIAN_MIN_SPEED = 15
THRESHOLD = 25
BRAKE_ACTION = 1.0
SAFETY_DISTANCE = 25

behavior EgoBehavior(trajectory):
    try:
        do FollowTrajectoryBehavior(target_speed=7, trajectory=trajectory)
        terminate

    interrupt when withinDistanceToObjsInLane(self, SAFETY_DISTANCE):
        take SetBrakeAction(BRAKE_ACTION)

behavior PedestrianBehavior(speed=1, threshold=15):
    do CrossingBehavior(ego, speed, threshold)

## DEFINING SPATIAL RELATIONS
# make sure to put '*' to uniformly randomly select from all elements of the list
intersec = Uniform(*network.intersections)
startLane = Uniform(*intersec.incomingLanes)

# Get the ego manuever
maneuvers = filter(lambda i: i.type != ManeuverType.STRAIGHT, startLane.maneuvers)
maneuver = Uniform(*maneuvers)
trajectory = [maneuver.startLane, maneuver.connectingLane, maneuver.endLane]

spot = new OrientedPoint in maneuver.startLane.centerline
ego = new Car at spot,
    with blueprint EGO_MODEL,
    with behavior EgoBehavior(trajectory=trajectory)

spotPedestrian = new OrientedPoint in maneuver.endLane.centerline,
    facing roadDirection
Pedestrian = new Pedestrian at spotPedestrian offset by 3.5@0,
    with heading 90 deg relative to spotPedestrian.heading,
    with behavior PedestrianBehavior(PEDESTRIAN_MIN_SPEED, THRESHOLD),
    with regionContainedIn None

require (maneuver.endLane.sections[0]._slowerLane is None)
require 10 <= (distance to intersec) <= 15
require 10 <= (distance from Pedestrian to intersec) <= 20
terminate when (distance to intersec) >= 30
