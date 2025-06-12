### Stop‑sign intersection scenario with background traffic (CARLA UE4)

# MAP & SIMULATOR --------------------------------------------------------------
param map = localPath('assets/Town03.xodr')
param carla_map = 'Town03'
model srunner.scenic.models.model

# BLUEPRINTS -------------------------------------------------------------------
param ego_blueprint_param = "vehicle.lincoln.mkz_2017"
param adv_blueprint_param = "vehicle.dodge.charger_police"
param bg_blueprints = [
    "vehicle.bmw.grandtourer", "vehicle.chevrolet.impala",
    "vehicle.micro.microlino", "vehicle.audi.a2"]

# SPEEDS & DISTANCES -----------------------------------------------------------
param ego_speed = 8
param adv_speed = 12
param safety_distance = 20

param ego_dist_min = 15
param ego_dist_max = 20
param adv_dist_min = 15
param adv_dist_max = 20

param stop_distance = 3
param stop_time = 2
param sync_delay = 2
param collision_advance = 2

EGO_DIST_MIN = globalParameters.ego_dist_min
EGO_DIST_MAX = globalParameters.ego_dist_max
ADV_DIST_MIN = globalParameters.adv_dist_min
ADV_DIST_MAX = globalParameters.adv_dist_max

EGO_DIST_MIN = globalParameters.stop_distance-1
EGO_DIST_MAX = globalParameters.stop_distance+1
ADV_DIST_MIN = 10
ADV_DIST_MAX = 15

# BEHAVIORS --------------------------------------------------------------------
behavior EgoBehavior(speed, traj, intersection):

    # do FollowTrajectoryBehavior(target_speed = speed, trajectory = traj) until (distance to intersection < globalParameters.stop_distance)
    do ForeverBrake(1.0) for globalParameters.stop_time seconds 
    try:
        do FollowTrajectoryBehavior(target_speed = speed, trajectory = traj)
    interrupt when withinDistanceToAnyObjs(ego, globalParameters.safety_distance):
        take SetBrakeAction(1.0)
    terminate

behavior AdvRunStopBehavior(intersection, traj, collision_region):

    # do SynchronizeIntersectionEntry(globalParameters.sync_delay, intersection, traj)
    do SynchronizeCollision(globalParameters.collision_advance, collision_region, traj)
    do FollowTrajectoryBehavior(10, traj)
    do FollowLaneBehavior(10)

# ROAD‑NETWORK LOGIC -----------------------------------------------------------
four_way_stops = filter(lambda i: len(i.roads) >= 3 and not i.isSignalized, network.intersections)
intersection = Uniform(*four_way_stops)

# PRIMARY VEHICLES -------------------------------------------------------------

ego_start_lane = Uniform(*intersection.incomingLanes)
ego_maneuver = Uniform(*filter(lambda m: m.type == ManeuverType.STRAIGHT, ego_start_lane.maneuvers))
ego_traj = [ego_maneuver.startLane, ego_maneuver.connectingLane, ego_maneuver.endLane]
ego_spawn = new OrientedPoint in ego_maneuver.startLane.centerline

ego = new Car at ego_spawn,
    with blueprint globalParameters.ego_blueprint_param,
    with behavior EgoBehavior(globalParameters.ego_speed, ego_traj, intersection)

adv_maneuver = Uniform(*ego_maneuver.conflictingManeuvers)
adv_traj = [adv_maneuver.startLane, adv_maneuver.connectingLane, adv_maneuver.endLane]
adv_spawn = new OrientedPoint in adv_maneuver.startLane.centerline

collision_region = ego_maneuver.connectingLane.intersect(adv_maneuver.connectingLane)

adversary = new Car at adv_spawn,
    with blueprint globalParameters.adv_blueprint_param,
    with behavior AdvRunStopBehavior(intersection, adv_traj, collision_region)

# BACKGROUND TRAFFIC -----------------------------------------------------------

## Spawn X amount of vehicles
# def SpawnBGActor():
#     bg_lane = Uniform(*intersection.incomingLanes)
#     bg_maneuver = Uniform(*bg_lane.maneuvers)
#     bg_traj = [bg_maneuver.startLane, bg_maneuver.connectingLane, bg_maneuver.endLane]
#     bg_spawn = new OrientedPoint in bg_maneuver.startLane.centerline
#     bg_car = new Car at bg_spawn,
#         with blueprint Uniform(*globalParameters.bg_blueprints),
#         with behavior BGFollowLane(globalParameters.bg_speed)
#     return bg_car
# bgCars = [SpawnBGActor() for _ in range(globalParameters.bg_vehicles)]


### Get the side lane from the ego
# if ego.laneSection._slowerLane is not None:
#     bgCars.append(SpawnBGActor(ego.laneSection._slowerLane.lane))


### Get the adjacent lanes
# for lanesection in ego.laneSection.adjacentLanes:
#     bgCars.append(SpawnBGActor(lanesection.lane))

# trail_adv_spawn = new OrientedPoint following roadDirection from adv_spawn for -Uniform(globalParameters.trailing_offset_min, globalParameters.trailing_offset_max)

# trailAdv = new Car at trail_adv_spawn,
#     with blueprint Uniform(*globalParameters.bg_blueprints),
#     with behavior BGFollowLane(globalParameters.bg_speed)

# opp_lane = ego_maneuver.startLane.oppositeLane
# opp_ref_point = new OrientedPoint in opp_lane.centerline
# opp_spawn = new OrientedPoint following roadDirection from opp_ref_point for Uniform(globalParameters.opp_offset_min, globalParameters.opp_offset_max)

# oppCar = new Car at opp_spawn,
#     with blueprint Uniform(*globalParameters.bg_blueprints),
#     with behavior BGFollowLane(globalParameters.bg_speed)

# REQUIREMENTS -----------------------------------------------------------------
require EGO_DIST_MIN <= (distance to intersection) <= EGO_DIST_MAX
require ADV_DIST_MIN <= (distance from adversary to intersection) <= ADV_DIST_MAX

# TERMINATION ------------------------------------------------------------------
terminate when (distance to ego_spawn) > 70
