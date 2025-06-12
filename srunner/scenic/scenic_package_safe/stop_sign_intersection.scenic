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
param bg_speed  = 6
param safety_distance = 20
param ego_brake_intensity = 1.0

param ego_dist_min = 15
param ego_dist_max = 25
param adv_dist_min = 15
param adv_dist_max = 25
param trailing_offset_min = 20
param trailing_offset_max = 30
param opp_offset_min      = 40
param opp_offset_max      = 50

EGO_DIST_MIN = globalParameters.ego_dist_min
EGO_DIST_MAX = globalParameters.ego_dist_max
ADV_DIST_MIN = globalParameters.adv_dist_min
ADV_DIST_MAX = globalParameters.adv_dist_max

# BEHAVIORS --------------------------------------------------------------------
behavior EgoBehavior(speed, traj):
    try:
        do FollowTrajectoryBehavior(target_speed = speed, trajectory = traj)
        do FollowLaneBehavior(target_speed = speed)
    interrupt when withinDistanceToAnyObjs(self, globalParameters.safety_distance):
        take SetBrakeAction(globalParameters.ego_brake_intensity)

behavior AdvRunStopBehavior(speed, traj):
    do FollowTrajectoryBehavior(target_speed = speed, trajectory = traj)

behavior BGFollowLane(speed):
    try:
        do FollowLaneBehavior(target_speed = speed)
    interrupt when withinDistanceToAnyObjs(self, globalParameters.safety_distance):
        take SetBrakeAction(1.0)

# ROAD‑NETWORK LOGIC -----------------------------------------------------------
four_way_stops = filter(lambda i: i.is4Way, network.intersections)
intersection = Uniform(*four_way_stops)

ego_start_lane = Uniform(*intersection.incomingLanes)
ego_maneuver = Uniform(*filter(lambda m: m.type == ManeuverType.STRAIGHT,
                               ego_start_lane.maneuvers))
ego_traj = [ego_maneuver.startLane, ego_maneuver.connectingLane, ego_maneuver.endLane]

adv_maneuver = Uniform(*filter(lambda m: m.type == ManeuverType.STRAIGHT,
                               ego_maneuver.conflictingManeuvers))
adv_traj = [adv_maneuver.startLane, adv_maneuver.connectingLane, adv_maneuver.endLane]

# PRIMARY VEHICLES -------------------------------------------------------------
ego_spawn = new OrientedPoint in ego_maneuver.startLane.centerline
adv_spawn = new OrientedPoint in adv_maneuver.startLane.centerline

ego = new Car at ego_spawn,
    with blueprint globalParameters.ego_blueprint_param,
    with behavior EgoBehavior(globalParameters.ego_speed, ego_traj)

adversary = new Car at adv_spawn,
    with blueprint globalParameters.adv_blueprint_param,
    with behavior AdvRunStopBehavior(globalParameters.adv_speed, adv_traj)

# BACKGROUND TRAFFIC -----------------------------------------------------------
trail_ego_spawn = new OrientedPoint following roadDirection from ego_spawn for -Uniform(globalParameters.trailing_offset_min, globalParameters.trailing_offset_max)

trailEgo = new Car at trail_ego_spawn,
    with blueprint Uniform(*globalParameters.bg_blueprints),
    with behavior BGFollowLane(globalParameters.bg_speed)

trail_adv_spawn = new OrientedPoint following roadDirection from adv_spawn for -Uniform(globalParameters.trailing_offset_min, globalParameters.trailing_offset_max)

trailAdv = new Car at trail_adv_spawn,
    with blueprint Uniform(*globalParameters.bg_blueprints),
    with behavior BGFollowLane(globalParameters.bg_speed)

# opp_lane = ego_maneuver.startLane.oppositeLane
# opp_ref_point = new OrientedPoint in opp_lane.centerline
# opp_spawn = new OrientedPoint following roadDirection from opp_ref_point for Uniform(globalParameters.opp_offset_min, globalParameters.opp_offset_max)

# oppCar = new Car at opp_spawn,
#     with blueprint Uniform(*globalParameters.bg_blueprints),
#     with behavior BGFollowLane(globalParameters.bg_speed)

# REQUIREMENTS -----------------------------------------------------------------
require EGO_DIST_MIN <= (distance from ego to intersection) <= EGO_DIST_MAX
require ADV_DIST_MIN <= (distance from adversary to intersection) <= ADV_DIST_MAX
# require minDistanceBetween([trailEgo, trailAdv, oppCar], [ego, adversary]) >= 10

# TERMINATION ------------------------------------------------------------------
terminate when (distance from ego to ego_spawn) > 50
