#!/usr/bin/env python

#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Scenario spawning elements to make the town dynamic and interesting
"""

import math
from collections import OrderedDict
import py_trees
import numpy as np

import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import AtomicBehavior
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.scenario_helper import get_same_dir_lanes, get_opposite_dir_lanes

JUNCTION_ENTRY = 'entry'
JUNCTION_MIDDLE = 'middle'
JUNCTION_EXIT = 'exit'
JUNCTION_INACTIVE = 'inactive'

EGO_JUNCTION = 'junction'
EGO_ROAD = 'road'

DEBUG_ROAD = 'road'
DEBUG_OPPOSITE = 'opposite'
DEBUG_JUNCTION = 'junction'
DEBUG_ENTRY = 'entry'
DEBUG_EXIT = 'exit'
DEBUG_CONNECT = 'connect'

DEBUG_SMALL = 'small'
DEBUG_MEDIUM = 'medium'
DEBUG_LARGE = 'large'

DEBUG_COLORS = {
    DEBUG_ROAD: carla.Color(0, 0, 255),      # Blue
    DEBUG_OPPOSITE: carla.Color(255, 0, 0),  # Red
    DEBUG_JUNCTION: carla.Color(0, 0, 0),   # Black
    DEBUG_ENTRY: carla.Color(255, 255, 0),   # Yellow
    DEBUG_EXIT: carla.Color(0, 255, 255),    # Teal
    DEBUG_CONNECT: carla.Color(0, 255, 0),   # Green
}

DEBUG_TYPE = {
    DEBUG_SMALL: [0.8, 0.1],
    DEBUG_MEDIUM: [0.5, 0.15],
    DEBUG_LARGE: [0.2, 0.2],
}

def draw_string(world, location, string='', debug_type=DEBUG_ROAD, persistent=False):
    """Utility function to draw debugging strings"""
    v_shift, _ = DEBUG_TYPE.get(DEBUG_SMALL)
    l_shift = carla.Location(z=v_shift)
    color = DEBUG_COLORS.get(debug_type, DEBUG_ROAD)
    life_time = 0.07 if not persistent else 100000
    world.debug.draw_string(location + l_shift, string, False, color, life_time)


def draw_point(world, location, point_type=DEBUG_SMALL, debug_type=DEBUG_ROAD, persistent=False):
    """Utility function to draw debugging points"""
    v_shift, size = DEBUG_TYPE.get(point_type, DEBUG_SMALL)
    l_shift = carla.Location(z=v_shift)
    color = DEBUG_COLORS.get(debug_type, DEBUG_ROAD)
    life_time = 0.07 if not persistent else 100000
    world.debug.draw_point(location + l_shift, size, color, life_time)


def get_lane_key(waypoint):
    """Returns a key corresponding to the waypoint lane. Equivalent to a 'Lane'
    object and used to compare waypoint lanes"""
    return '' if waypoint is None else get_road_key(waypoint) + '*' + str(waypoint.lane_id)


def get_road_key(waypoint):
    """Returns a key corresponding to the waypoint road. Equivalent to a 'Road'
    object and used to compare waypoint roads"""
    return '' if waypoint is None else str(waypoint.road_id)


class Source(object):

    """
    Source object to store its position and its responsible actors
    """

    def __init__(self, wp, actors, entry_lane_wp='', dist_to_ego=0, active=True):  # pylint: disable=invalid-name
        self.wp = wp  # pylint: disable=invalid-name
        self.actors = actors
        self.active = active

        # For road sources
        self.dist_to_ego = dist_to_ego

        # For junction sources
        self.entry_lane_wp = entry_lane_wp
        self.previous_lane_keys = []  # Source lane and connecting lanes of the previous junction


class Junction(object):

    """
    Junction object. Stores its topology as well as its state, when active
    """

    def __init__(self, junction, junction_id, route_entry_index=None, route_exit_index=None):
        # Topology
        self.junctions = [junction]
        self.id = junction_id  # pylint: disable=invalid-name
        self.route_entry_index = route_entry_index
        self.route_exit_index = route_exit_index
        self.exit_road_length = 0
        self.route_entry_keys = []
        self.route_exit_keys = []
        self.opposite_entry_keys = []
        self.opposite_exit_keys = []
        self.entry_wps = []
        self.exit_wps = []
        self.entry_directions = {'ref': [], 'opposite': [], 'left': [], 'right': []}
        self.exit_directions = {'ref': [], 'opposite': [], 'left': [], 'right': []}

        # State
        self.entry_sources = []
        self.exit_dict = OrderedDict()
        self.actor_dict = OrderedDict()

        # Scenario interactions
        self.scenario_info = {
            'direction': None,
            'remove_entries': False,
            'remove_middle': False,
            'remove_exits': False,
        }
        self.stop_entries = False

    def contains(self, other_junction):
        """Checks whether or not a carla.Junction is part of the class"""
        other_id = other_junction.id
        for junction in self.junctions:
            if other_id == junction.id:
                return True
        return False


class BackgroundActivity(BasicScenario):

    """
    Implementation of a scenario to spawn a set of background actors,
    and to remove traffic jams in background traffic

    This is a single ego vehicle scenario
    """

    def __init__(self, world, ego_vehicle, config, route, night_mode=False, debug_mode=False, timeout=0):
        """
        Setup all relevant parameters and create scenario
        """
        self._map = CarlaDataProvider.get_map()
        self.ego_vehicle = ego_vehicle
        self.route = route
        self.config = config
        self._night_mode = night_mode
        self.debug = debug_mode
        self.timeout = timeout  # Timeout of scenario in seconds

        super(BackgroundActivity, self).__init__("BackgroundActivity",
                                                 [ego_vehicle],
                                                 config,
                                                 world,
                                                 debug_mode,
                                                 terminate_on_failure=True,
                                                 criteria_enable=True)

    def _create_behavior(self):
        """
        Basic behavior do nothing, i.e. Idle
        """
        # Check if a vehicle is further than X, destroy it if necessary and respawn it
        return BackgroundBehavior(self.ego_vehicle, self.route, self._night_mode)

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        return []

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        pass


class BackgroundBehavior(AtomicBehavior):
    """
    Handles the background activity
    """

    def __init__(self, ego_actor, route, night_mode=False, debug=False, name="BackgroundBehavior"):
        """
        Setup class members
        """
        super(BackgroundBehavior, self).__init__(name)
        self.debug = debug
        self._map = CarlaDataProvider.get_map()
        self._world = CarlaDataProvider.get_world()
        timestep = self._world.get_snapshot().timestamp.delta_seconds
        self._tm = CarlaDataProvider.get_client().get_trafficmanager(
            CarlaDataProvider.get_traffic_manager_port())
        self._tm.global_percentage_speed_difference(0.0)
        self._night_mode = night_mode

        # Global variables
        self._ego_actor = ego_actor
        self._ego_state = EGO_ROAD
        self._route_index = 0
        self._get_route_data(route)

        self._spawn_vertical_shift = 0.2
        self._reuse_dist = 10  # When spawning actors, might reuse actors closer to this distance
        self._spawn_free_radius = 20  # Sources closer to the ego will not spawn actors
        self._fake_junction_ids = []
        self._fake_lane_pair_keys = []

        # Road variables
        self._road_dict = {}  # Dictionary lane key -> actor source
        self._road_checker_index = 0
        self._road_ego_key = ""

        self._road_front_vehicles = 3  # Amount of vehicles in front of the ego
        self._road_back_vehicles = 3  # Amount of vehicles behind the ego
        self._road_vehicle_dist = 8  # Distance road vehicles leave betweeen each other[m]
        self._road_spawn_dist = 11  # Initial distance between spawned road vehicles [m]
        self._road_new_sources_dist = 20  # Distance of the source to the start of the new lanes
        self._radius_increase_ratio = 1.8  # Meters the radius increases per m/s of the ego

        # TODO: Remove these 3 and refine road radiuses
        self._extra_radius = 0.0  # Extra distance to avoid the road behavior from blocking. 
        self._extra_radius_increase_ratio = 0.5 * timestep  # Distance the radius increases per tick (0.5 m/s)
        self._max_extra_radius = 10  # Max extra distance

        self._road_num_front_vehicles = self._road_front_vehicles  # Checks the real amount of actors in the front of the ego
        self._road_extra_front_actors = 0  # For cases where we want more space but not more vehicles
        self._road_time_delay = 3

        self._base_min_radius = 0
        self._base_max_radius = 0
        self._min_radius = 0
        self._max_radius = 0
        self._junction_detection_dist = 0
        self._get_road_radius()

        # Junction variables
        self._junctions = []
        self._active_junctions = []

        self._junction_sources_dist = 40  # Distance from the entry sources to the junction [m]
        self._junction_vehicle_dist = 8  # Distance junction vehicles leave betweeen each other[m]
        self._junction_spawn_dist = 10  # Initial distance between spawned junction vehicles [m]
        self._junction_sources_max_actors = 5  # Maximum vehicles alive at the same time per source

        # Opposite lane variables
        self._opposite_actors = []
        self._opposite_sources = []
        self._opposite_route_index = 0

        self._opposite_removal_dist = 30  # Distance at which actors are destroyed
        self._opposite_sources_dist = 60  # Distance from the ego to the opposite sources [m]
        self._opposite_vehicle_dist = 10  # Distance opposite vehicles leave betweeen each other[m]
        self._opposite_spawn_dist = 20  # Initial distance between spawned opposite vehicles [m]
        self._opposite_sources_max_actors = 8  # Maximum vehicles alive at the same time per source

        # Scenario 2 variables
        self._stopped_road_actors = []

        # Scenario 4 variables
        self._is_crossing_scenario_active = False
        self._crossing_scenario_actors = []
        self._ego_exitted_junction = False
        self._crossing_dist = None  # Distance between the crossing object and the junction exit
        self._start_ego_wp = None

        # Junction scenario variables
        self.scenario_info = {
            'direction': None,
            'remove_entries': False,
            'remove_middle': False,
            'remove_exits': False,
        }  # Same as the Junction.scenario_info, but this stores the data in case no junctions are active

        self._route_sources_active = True

    def _get_route_data(self, route):
        """Extract the information from the route"""
        self._route = []  # Transform the route into a list of waypoints
        self._accum_dist = []  # Save the total traveled distance for each waypoint
        prev_trans = None
        for trans, _ in route:
            self._route.append(self._map.get_waypoint(trans.location))
            if prev_trans:
                dist = trans.location.distance(prev_trans.location)
                self._accum_dist.append(dist + self._accum_dist[-1])
            else:
                self._accum_dist.append(0)
            prev_trans = trans

        self._route_length = len(route)
        self._route_index = 0
        self._route_buffer = 3

    def _get_road_radius(self):
        """
        Computes the min and max radius of the road behaviorm which will determine the speed of the vehicles.
        Vehicles closer than the min radius maintain full speed, while those further than max radius are
        stopped. Between the two, the velocity decreases linearly"""
        self._base_min_radius = (self._road_num_front_vehicles + self._road_extra_front_actors) * self._road_spawn_dist
        self._base_max_radius = (self._road_num_front_vehicles + self._road_extra_front_actors + 1) * self._road_spawn_dist
        self._min_radius = self._base_min_radius
        self._max_radius = self._base_max_radius

    def initialise(self):
        """Creates the background activity actors. Pressuposes that the ego is at a road"""
        self._create_junction_dict()
        ego_wp = self._route[0]
        self._road_ego_key = get_lane_key(ego_wp)
        same_dir_wps = get_same_dir_lanes(ego_wp)

        self._initialise_road_behavior(same_dir_wps)
        self._initialise_opposite_sources()
        self._initialise_road_checker()

    def update(self):
        prev_ego_index = self._route_index

        # Check if the TM destroyed an actor
        if self._route_index > 0: # TODO: This check is due to intialization problem.
            self._check_background_actors()

        # Update ego's route position. For robustness, the route point is used for most calculus
        self._update_ego_route_location()

        # Parameters and scenarios
        self._update_parameters()

        # Update ego state.
        if self._ego_state == EGO_JUNCTION:
            self._monitor_ego_junction_exit()
        self._monitor_incoming_junctions()

        # Update_actors
        if self._ego_state == EGO_JUNCTION:
            self._update_junction_actors()
            self._update_junction_sources()
        else:
            self._update_road_actors()
            self._update_road_sources(prev_ego_index)
            self._move_road_checker(prev_ego_index)
            self._move_opposite_sources(prev_ego_index)
            self._monitor_road_changes(prev_ego_index)
            self._update_opposite_sources()

        # Update non junction sources.
        self._update_opposite_actors()

        self._monitor_crossing_scenario_end()

        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        """Destroy all actors"""
        all_actors = self._get_actors()
        for actor in list(all_actors):
            self._destroy_actor(actor)
        super(BackgroundBehavior, self).terminate(new_status)

    def _get_actors(self):
        """Returns a list of all actors part of the background activity"""
        actors = list(self._opposite_actors)
        for lane in self._road_dict:
            actors.extend(self._road_dict[lane].actors)
        for junction in self._active_junctions:
            actors.extend(list(junction.actor_dict))
        return actors

    def _check_background_actors(self):
        """Checks if the Traffic Manager has removed a backgroudn actor"""
        background_actors = self._get_actors()
        alive_ids = [actor.id for actor in self._world.get_actors().filter('vehicle*')]
        for actor in background_actors:
            if actor.id not in alive_ids:
                self._remove_actor_info(actor)

    ################################
    ##       Junction cache       ##
    ################################

    def _create_junction_dict(self):
        """Extracts the junctions the ego vehicle will pass through."""
        data = self._get_junctions_data()
        fake_data, filtered_data = self._filter_fake_junctions(data)
        self._get_fake_lane_pairs(fake_data)
        route_data = self._join_complex_junctions(filtered_data)
        self._add_junctions_topology(route_data)
        self._junctions = route_data

    def _get_junctions_data(self):
        """Gets all the junctions the ego passes through"""
        junction_data = []
        junction_num = 0
        start_index = 0

        # Ignore the junction the ego spawns at
        for i in range(0, self._route_length - 1):
            if not self._is_junction(self._route[i]):
                start_index = i
                break

        for i in range(start_index, self._route_length - 1):
            next_wp = self._route[i+1]
            prev_junction = junction_data[-1] if len(junction_data) > 0 else None

            # Searching for the junction exit
            if prev_junction and prev_junction.route_exit_index is None:
                if not self._is_junction(next_wp) or next_wp.get_junction().id != junction_id:
                    prev_junction.route_exit_index = i+1

            # Searching for a junction
            elif self._is_junction(next_wp):
                junction_id = next_wp.get_junction().id
                if prev_junction:
                    start_dist = self._accum_dist[i]
                    prev_end_dist = self._accum_dist[prev_junction.route_exit_index]
                    prev_junction.exit_road_length = start_dist - prev_end_dist

                # Same junction as the prev one and closer than 2 meters
                if prev_junction and prev_junction.junctions[-1].id == junction_id:
                    start_dist = self._accum_dist[i]
                    prev_end_dist = self._accum_dist[prev_junction.route_exit_index]
                    distance = start_dist - prev_end_dist
                    if distance < 2:
                        prev_junction.junctions.append(next_wp.get_junction())
                        prev_junction.route_exit_index = None
                        continue

                junction_data.append(Junction(next_wp.get_junction(), junction_num, i))
                junction_num += 1

        if len(junction_data) > 0:
            road_end_dist = self._accum_dist[self._route_length - 1]
            if junction_data[-1].route_exit_index:
                route_start_dist = self._accum_dist[junction_data[-1].route_exit_index]
            else:
                route_start_dist = self._accum_dist[self._route_length - 1]
            junction_data[-1].exit_road_length = road_end_dist - route_start_dist

        return junction_data

    def _filter_fake_junctions(self, data):
        """
        Filters fake junctions. As a general note, a fake junction is that where no road lane divide in two.
        However, this might fail for some CARLA maps, so check junctions which have all lanes straight too
        """
        fake_data = []
        filtered_data = []
        threshold = math.radians(15)

        for junction_data in data:
            used_entry_lanes = []
            used_exit_lanes = []
            for junction in junction_data.junctions:
                for entry_wp, exit_wp in junction.get_waypoints(carla.LaneType.Driving):
                    entry_wp = self._get_junction_entry_wp(entry_wp)
                    if not entry_wp:
                        continue
                    if get_lane_key(entry_wp) not in used_entry_lanes:
                        used_entry_lanes.append(get_lane_key(entry_wp))

                    exit_wp = self._get_junction_exit_wp(exit_wp)
                    if not exit_wp:
                        continue
                    if get_lane_key(exit_wp) not in used_exit_lanes:
                        used_exit_lanes.append(get_lane_key(exit_wp))

            if not used_entry_lanes and not used_exit_lanes:
                fake_data.append(junction_data)
                continue

            found_turn = False
            for entry_wp, exit_wp in junction_data.junctions[0].get_waypoints(carla.LaneType.Driving):
                entry_heading = entry_wp.transform.get_forward_vector()
                exit_heading = exit_wp.transform.get_forward_vector()
                dot = entry_heading.x * exit_heading.x + entry_heading.y * exit_heading.y
                if dot < math.cos(threshold):
                    found_turn = True
                    break

            if not found_turn:
                fake_data.append(junction_data)
            else:
                filtered_data.append(junction_data)

        return fake_data, filtered_data

    def _get_complex_junctions(self):
        """
        Function to hardcode the topology of some complex junctions. This is done for the roundabouts,
        as the current API doesn't offer that info as well as others such as the gas station at Town04.
        If there are micro lanes between connected junctions, add them to the fake_lane_keys, connecting
        them when their topology is calculated
        """
        complex_junctions = []
        fake_lane_keys = []

        if 'Town03' in self._map.name:
            # Roundabout, take it all as one
            complex_junctions.append([
                self._map.get_waypoint_xodr(1100, -5, 16.6).get_junction(),
                self._map.get_waypoint_xodr(1624, -5, 25.3).get_junction(),
                self._map.get_waypoint_xodr(1655, -5, 8.3).get_junction(),
                self._map.get_waypoint_xodr(1772, 3, 16.2).get_junction(),
                self._map.get_waypoint_xodr(1206, -5, 5.9).get_junction()])
            fake_lane_keys.extend([
                ['37*-4', '36*-4'], ['36*-4', '37*-4'],
                ['37*-5', '36*-5'], ['36*-5', '37*-5'],
                ['38*-4', '12*-4'], ['12*-4', '38*-4'],
                ['38*-5', '12*-5'], ['12*-5', '38*-5']])

            # Gas station
            complex_junctions.append([
                self._map.get_waypoint_xodr(1031, -1, 11.3).get_junction(),
                self._map.get_waypoint_xodr(100, -1, 18.8).get_junction(),
                self._map.get_waypoint_xodr(1959, -1, 22.7).get_junction()])
            fake_lane_keys.extend([
                ['32*-2', '33*-2'], ['33*-2', '32*-2'],
                ['32*-1', '33*-1'], ['33*-1', '32*-1'],
                ['32*4', '33*4'], ['33*4', '32*4'],
                ['32*5', '33*5'], ['33*5', '32*5']])

        elif 'Town04' in self._map.name:
            # Gas station
            complex_junctions.append([
                self._map.get_waypoint_xodr(518, -1, 8.1).get_junction(),
                self._map.get_waypoint_xodr(886, 1, 10.11).get_junction(),
                self._map.get_waypoint_xodr(467, 1, 25.8).get_junction()])

        self._fake_lane_pair_keys.extend(fake_lane_keys)
        return complex_junctions

    def _join_complex_junctions(self, filtered_data):
        """
        Joins complex junctions into one. This makes it such that all the junctions,
        as well as their connecting lanes, are treated as the same junction
        """
        route_data = []
        prev_index = -1

        # If entering a complex, add all its junctions to the list
        for junction_data in filtered_data:
            junction = junction_data.junctions[0]
            prev_junction = route_data[-1] if len(route_data) > 0 else None
            complex_junctions = self._get_complex_junctions()

            # Get the complex index
            current_index = -1
            for i, complex_junctions in enumerate(complex_junctions):
                complex_ids = [j.id for j in complex_junctions]
                if junction.id in complex_ids:
                    current_index = i
                    break

            if current_index == -1:
                # Outside a complex, add it
                route_data.append(junction_data)

            elif current_index == prev_index:
                # Same complex as the previous junction
                prev_junction.route_exit_index = junction_data.route_exit_index

            else:
                # New complex, add it
                junction_ids = [j.id for j in junction_data.junctions]
                for complex_junction in complex_junctions:
                    if complex_junction.id not in junction_ids:
                        junction_data.junctions.append(complex_junction)

                route_data.append(junction_data)

            prev_index = current_index

        return route_data

    def _get_fake_lane_pairs(self, fake_data):
        """Gets a list of entry-exit lanes of the fake junctions"""
        for fake_junctions_data in fake_data:
            for junction in fake_junctions_data.junctions:
                for entry_wp, exit_wp in junction.get_waypoints(carla.LaneType.Driving):
                    while self._is_junction(entry_wp):
                        entry_wps = entry_wp.previous(0.5)
                        if len(entry_wps) == 0:
                            break  # Stop when there's no prev
                        entry_wp = entry_wps[0]
                    if self._is_junction(entry_wp):
                        continue  # Triggered by the loops break

                    while self._is_junction(exit_wp):
                        exit_wps = exit_wp.next(0.5)
                        if len(exit_wps) == 0:
                            break  # Stop when there's no prev
                        exit_wp = exit_wps[0]
                    if self._is_junction(exit_wp):
                        continue  # Triggered by the loops break

                    self._fake_junction_ids.append(junction.id)
                    self._fake_lane_pair_keys.append([get_lane_key(entry_wp), get_lane_key(exit_wp)])

    def _get_junction_entry_wp(self, entry_wp):
        """For a junction waypoint, returns a waypoint outside of it that entrys into its lane"""
        # Exit the junction
        while self._is_junction(entry_wp):
            entry_wps = entry_wp.previous(0.2)
            if len(entry_wps) == 0:
                return None  # Stop when there's no prev
            entry_wp = entry_wps[0]
        return entry_wp

    def _get_junction_exit_wp(self, exit_wp):
        """For a junction waypoint, returns a waypoint outside of it from which the lane exits the junction"""
        while self._is_junction(exit_wp):
            exit_wps = exit_wp.next(0.2)
            if len(exit_wps) == 0:
                return None  # Stop when there's no prev
            exit_wp = exit_wps[0]
        return exit_wp

    def _get_closest_junction_waypoint(self, waypoint, junction_wps):
        """
        Matches a given wp to another one inside the list.
        This is first done by checking its key, and if this fails, the closest wp is chosen
        """
        # Check the lane keys
        junction_keys = [get_lane_key(waypoint_) for waypoint_ in junction_wps]
        if get_lane_key(waypoint) in junction_keys:
            return waypoint

        # Get the closest one
        closest_dist = float('inf')
        closest_junction_wp = None
        route_location = waypoint.transform.location
        for junction_wp in junction_wps:
            distance = junction_wp.transform.location.distance(route_location)
            if distance < closest_dist:
                closest_dist = distance
                closest_junction_wp = junction_wp

        return closest_junction_wp

    def _is_route_wp_behind_junction_wp(self, route_wp, junction_wp):
        """Checks if an actor is behind the ego. Uses the route transform"""
        route_location = route_wp.transform.location
        junction_transform = junction_wp.transform
        junction_heading = junction_transform.get_forward_vector()
        wps_vec = route_location - junction_transform.location
        if junction_heading.x * wps_vec.x + junction_heading.y * wps_vec.y < - 0.09:  # 85º
            return True
        return False

    def _add_junctions_topology(self, route_data):
        """Gets the entering and exiting lanes of a multijunction"""
        for junction_data in route_data:
            used_entry_lanes = []
            used_exit_lanes = []
            entry_lane_wps = []
            exit_lane_wps = []

            if self.debug:
                print(' --------------------- ')
            for junction in junction_data.junctions:
                for entry_wp, exit_wp in junction.get_waypoints(carla.LaneType.Driving):

                    entry_wp = self._get_junction_entry_wp(entry_wp)
                    if not entry_wp:
                        continue
                    if get_lane_key(entry_wp) not in used_entry_lanes:
                        used_entry_lanes.append(get_lane_key(entry_wp))
                        entry_lane_wps.append(entry_wp)
                        if self.debug:
                            draw_point(self._world, entry_wp.transform.location, DEBUG_SMALL, DEBUG_ENTRY, True)

                    exit_wp = self._get_junction_exit_wp(exit_wp)
                    if not exit_wp:
                        continue
                    if get_lane_key(exit_wp) not in used_exit_lanes:
                        used_exit_lanes.append(get_lane_key(exit_wp))
                        exit_lane_wps.append(exit_wp)
                        if self.debug:
                            draw_point(self._world, exit_wp.transform.location, DEBUG_SMALL, DEBUG_EXIT, True)

            # Check for connecting lanes. This is pretty much for the roundabouts, but some weird geometries
            # make it possible for single junctions to have the same road entering and exiting. Two cases,
            # Lanes that exit one junction and enter another (or viceversa)
            exit_lane_keys = [get_lane_key(wp) for wp in exit_lane_wps]
            entry_lane_keys = [get_lane_key(wp) for wp in entry_lane_wps]
            for wp in list(entry_lane_wps):
                if get_lane_key(wp) in exit_lane_keys:
                    entry_lane_wps.remove(wp)
                    if self.debug:
                        draw_point(self._world, wp.transform.location, DEBUG_SMALL, DEBUG_CONNECT, True)

            for wp in list(exit_lane_wps):
                if get_lane_key(wp) in entry_lane_keys:
                    exit_lane_wps.remove(wp)
                    if self.debug:
                        draw_point(self._world, wp.transform.location, DEBUG_SMALL, DEBUG_CONNECT, True)

            # Lanes with a fake junction in the middle (maps junction exit to fake junction entry and viceversa)
            for entry_key, exit_key in self._fake_lane_pair_keys:
                entry_wp = None
                for wp in entry_lane_wps:
                    if get_lane_key(wp) == exit_key:  # A junction exit is a fake junction entry
                        entry_wp = wp
                        break
                exit_wp = None
                for wp in exit_lane_wps:
                    if get_lane_key(wp) == entry_key:  # A junction entry is a fake junction exit
                        exit_wp = wp
                        break
                if entry_wp and exit_wp:
                    entry_lane_wps.remove(entry_wp)
                    exit_lane_wps.remove(exit_wp)
                    if self.debug:
                        draw_point(self._world, entry_wp.transform.location, DEBUG_SMALL, DEBUG_CONNECT, True)
                        draw_point(self._world, exit_wp.transform.location, DEBUG_SMALL, DEBUG_CONNECT, True)

            junction_data.entry_wps = entry_lane_wps
            junction_data.exit_wps = exit_lane_wps
            for exit_wp in exit_lane_wps:
                junction_data.exit_dict[get_lane_key(exit_wp)] = {
                    'actors': [],
                    'max_actors': 0,
                    'ref_wp': None,
                    'max_distance': 0,
                }

            # Filter the entries and exits that correspond to the route
            route_entry_wp = self._route[junction_data.route_entry_index]

            # Junction entry
            for wp in get_same_dir_lanes(route_entry_wp):
                junction_wp = self._get_closest_junction_waypoint(wp, entry_lane_wps)
                junction_data.route_entry_keys.append(get_lane_key(junction_wp))
            for wp in get_opposite_dir_lanes(route_entry_wp):
                junction_wp = self._get_closest_junction_waypoint(wp, exit_lane_wps)
                junction_data.opposite_exit_keys.append(get_lane_key(junction_wp))

            # Junction exit
            if junction_data.route_exit_index:  # Can be None if route ends at a junction
                route_exit_wp = self._route[junction_data.route_exit_index]
                for wp in get_same_dir_lanes(route_exit_wp):
                    junction_wp = self._get_closest_junction_waypoint(wp, exit_lane_wps)
                    junction_data.route_exit_keys.append(get_lane_key(junction_wp))
                for wp in get_opposite_dir_lanes(route_exit_wp):
                    junction_wp = self._get_closest_junction_waypoint(wp, entry_lane_wps)
                    junction_data.opposite_entry_keys.append(get_lane_key(junction_wp))

            # Add the entry directions of each lane with respect to the route. Used for scenarios 7 to 9
            route_entry_yaw = route_entry_wp.transform.rotation.yaw
            for wp in entry_lane_wps:
                diff = (wp.transform.rotation.yaw - route_entry_yaw) % 360
                if diff > 330.0:
                    direction = 'ref'
                elif diff > 225.0:
                    direction = 'right'
                elif diff > 135.0:
                    direction = 'opposite'
                elif diff > 30.0:
                    direction = 'left'
                else:
                    direction = 'ref'

                junction_data.entry_directions[direction].append(get_lane_key(wp))

            # Supposing scenario vehicles go straight, these correspond to the exit lanes of the entry directions
            for wp in exit_lane_wps:
                diff = (wp.transform.rotation.yaw - route_entry_yaw) % 360
                if diff > 330.0:
                    direction = 'ref'
                elif diff > 225.0:
                    direction = 'right'
                elif diff > 135.0:
                    direction = 'opposite'
                elif diff > 30.0:
                    direction = 'left'
                else:
                    direction = 'ref'

                junction_data.exit_directions[direction].append(get_lane_key(wp))

            if self.debug:
                exit_lane = self._route[junction_data.route_exit_index] if junction_data.route_exit_index else None
                print('> R Entry Lane: {}'.format(get_lane_key(self._route[junction_data.route_entry_index])))
                print('> R Exit  Lane: {}'.format(get_lane_key(exit_lane)))
                entry_print = '> J Entry Lanes: '
                for entry_wp in entry_lane_wps:
                    key = get_lane_key(entry_wp)
                    entry_print += key + ' ' * (6 - len(key))
                print(entry_print)
                exit_print = '> J Exit  Lanes: '
                for exit_wp in exit_lane_wps:
                    key = get_lane_key(exit_wp)
                    exit_print += key + ' ' * (6 - len(key))
                print(exit_print)
                route_entry = '> R-J Entry Lanes: '
                for entry_key in junction_data.route_entry_keys:
                    route_entry += entry_key + ' ' * (6 - len(entry_key))
                print(route_entry)
                route_exit = '> R-J Route Exit  Lanes: '
                for exit_key in junction_data.route_exit_keys:
                    route_exit += exit_key + ' ' * (6 - len(exit_key))
                print(route_exit)
                route_oppo_entry = '> R-J Oppo Entry Lanes: '
                for oppo_entry_key in junction_data.opposite_entry_keys:
                    route_oppo_entry += oppo_entry_key + ' ' * (6 - len(oppo_entry_key))
                print(route_oppo_entry)
                route_oppo_exit = '> R-J Oppo Exit  Lanes: '
                for oppo_exit_key in junction_data.opposite_exit_keys:
                    route_oppo_exit += oppo_exit_key + ' ' * (6 - len(oppo_exit_key))
                print(route_oppo_exit)

    def _is_junction(self, waypoint):
        if not waypoint.is_junction or waypoint.junction_id in self._fake_junction_ids:
            return False
        return True

    ################################
    ##       Mode functions       ##
    ################################

    def _add_actor_dict_element(self, actor_dict, actor, exit_lane_key='', at_oppo_entry_lane=False):
        """
        Adds a new actor to the actor dictionary.
        'exit_lane_key' is used to know at which exit lane (if any) is the vehicle
        'at_oppo_entry_lane' whether or not the actor is part of the entry at the opposite lane the route exits through.
        This will be the ones that aren't removed
        """
        actor_dict[actor] = {
            'state': JUNCTION_ENTRY if not exit_lane_key else JUNCTION_EXIT,
            'exit_lane_key': exit_lane_key,  
            'at_oppo_entry_lane': at_oppo_entry_lane
        }

    def _switch_to_junction_mode(self, junction):
        """
        Prepares the junction mode, removing all road behaviours.
        Actors that are stopped via a scenario will still wait.
        """
        self._ego_state = EGO_JUNCTION
        for lane in self._road_dict:
            for actor in self._road_dict[lane].actors:
                # TODO: Map the actors to the junction entry to have full control of them. This should remove the 'at_oppo_entry_lane'
                self._add_actor_dict_element(junction.actor_dict, actor)
                if actor not in self._stopped_road_actors:
                    self._tm.vehicle_percentage_speed_difference(actor, 0)

        for lane_key in self._road_dict:
            source = self._road_dict[lane_key]
            if get_lane_key(source.wp) in junction.route_entry_keys:
                junction.entry_sources.append(Source(
                    source.wp, source.actors, entry_lane_wp=source.wp, active=self._route_sources_active)
                )
            # TODO: Else should map the source to the entry and add it

        self._road_dict.clear()
        self._road_num_front_vehicles = self._road_front_vehicles
        self._opposite_sources.clear()

    def _initialise_junction_scenario(self, direction, remove_entries, remove_exits, remove_middle):
        """
        Removes all vehicles in a particular 'direction' as well as all actors inside the junction.
        Additionally, activates some flags to ensure the junction is empty at all times
        """
        if self._active_junctions:
            scenario_junction = self._active_junctions[0]
            scenario_junction.scenario_info = {
                'direction': direction,
                'remove_entries': remove_entries,
                'remove_middle': remove_middle,
                'remove_exits': remove_exits,
            }
            entry_direction_keys = scenario_junction.entry_directions[direction]
            actor_dict = scenario_junction.actor_dict

            if remove_entries:
                for entry_source in scenario_junction.entry_sources:
                    if get_lane_key(entry_source.entry_lane_wp) in entry_direction_keys:
                        # Source is affected
                        actors = entry_source.actors
                        for actor in list(actors):
                            if actor_dict[actor]['state'] == JUNCTION_ENTRY:
                                # Actor is at the entry lane
                                self._destroy_actor(actor)

            if remove_exits:
                for exit_dir in scenario_junction.exit_directions[direction]:
                    for actor in list(scenario_junction.exit_dict[exit_dir]['actors']):
                        self._destroy_actor(actor)

            if remove_middle:
                actor_dict = scenario_junction.actor_dict
                for actor in list(actor_dict):
                    if actor_dict[actor]['state'] == JUNCTION_MIDDLE:
                        self._destroy_actor(actor)

        elif self._junctions:
            self._junctions[0].scenario_info = {
                'direction': direction,
                'remove_entries': remove_entries,
                'remove_middle': remove_middle,
                'remove_exits': remove_exits,
            }

    def _handle_junction_scenario_end(self, junction):
        """Ends the junction scenario interaction. This is pretty much useless as the junction
        scenario ends at the same time as the active junction, but in the future it might not"""
        junction.scenario_info = {
            'direction': None,
            'remove_entries': False,
            'remove_middle': False,
            'remove_exits': False,
        }

    def _monitor_crossing_scenario_end(self):
        """Monitors the ego distance to the junction to know if the scenario 4 has ended"""
        if self._ego_exitted_junction:
            ref_location = self._start_ego_wp.transform.location
            ego_location = self._ego_wp.transform.location
            if ego_location.distance(ref_location) > self._crossing_dist:
                for actor in self._crossing_scenario_actors:
                    self._tm.vehicle_percentage_speed_difference(actor, 0)
                self._is_crossing_scenario_active = False
                self._crossing_scenario_actors.clear()
                self._ego_exitted_junction = False
                self._crossing_dist = None

    def _handle_crossing_scenario_interaction(self, junction, ego_wp):
        """
        Handles the interation between the scenario 4 of the Leaderboard and the background activity.
        This removes all vehicles near the bycicle path, and stops the others so that they don't interfere
        """
        if not self._is_crossing_scenario_active:
            return

        self._ego_exitted_junction = True
        self._start_ego_wp = ego_wp
        min_crossing_space = 2

        # Actor exitting the junction
        exit_dict = junction.exit_dict
        for exit_key in exit_dict:
            if exit_key not in junction.route_exit_keys:
                continue
            actors = exit_dict[exit_key]['actors']
            exit_lane_wp = exit_dict[exit_key]['ref_wp']
            exit_lane_location = exit_lane_wp.transform.location
            for actor in list(actors):
                actor_location = CarlaDataProvider.get_location(actor)
                if not actor_location:
                    self._destroy_actor(actor)
                    continue

                dist_to_scenario = exit_lane_location.distance(actor_location) - self._crossing_dist
                actor_length = actor.bounding_box.extent.x
                if abs(dist_to_scenario) < actor_length + min_crossing_space:
                    self._destroy_actor(actor)
                    continue

                if dist_to_scenario > 0:
                    continue  # Don't stop the actors that have already passed the scenario

                if get_lane_key(ego_wp) == get_lane_key(exit_lane_wp):
                    self._destroy_actor(actor)
                    continue  # Actor at the ego lane and between the ego and scenario

                self._crossing_scenario_actors.append(actor)

        # Actor entering the junction
        for entry_source in junction.entry_sources:
            entry_lane_wp = entry_source.entry_lane_wp
            if get_lane_key(entry_lane_wp) in junction.opposite_entry_keys:
                # Source is affected
                actors = entry_source.actors
                entry_lane_location = entry_lane_wp.transform.location
                for actor in list(actors):
                    actor_location = CarlaDataProvider.get_location(actor)
                    if not actor_location:
                        self._destroy_actor(actor)
                        continue

                    crossing_space = abs(entry_lane_location.distance(actor_location) - self._crossing_dist)
                    actor_length = actor.bounding_box.extent.x
                    if crossing_space < actor_length + min_crossing_space:
                        self._destroy_actor(actor)
                        continue  # Actors blocking the path of the crossing obstacle

                    self._crossing_scenario_actors.append(actor)

        # Actors entering the next junction
        if len(self._active_junctions) > 1:
            next_junction = self._active_junctions[1]
            actors_dict = next_junction.actor_dict
            for actor in list(actors_dict):
                if actors_dict[actor]['state'] != JUNCTION_ENTRY:
                    continue

                actor_location = CarlaDataProvider.get_location(actor)
                if not actor_location:
                    self._destroy_actor(actor)
                    continue

                dist_to_scenario = exit_lane_location.distance(actor_location) - self._crossing_dist
                actor_length = actor.bounding_box.extent.x
                if abs(dist_to_scenario) < actor_length + min_crossing_space:
                    self._destroy_actor(actor)
                    continue

                if dist_to_scenario > 0:
                    continue  # Don't stop the actors that have already passed the scenario

                actor_wp = self._map.get_waypoint(actor_location)
                if get_lane_key(ego_wp) == get_lane_key(actor_wp):
                    self._destroy_actor(actor)
                    continue  # Actor at the ego lane and between the ego and scenario

                self._crossing_scenario_actors.append(actor)

        # Immediately freeze the actors
        for actor in self._crossing_scenario_actors:
            try:
                actor.set_target_velocity(carla.Vector3D(0, 0, 0))
                self._tm.vehicle_percentage_speed_difference(actor, 100)
            except RuntimeError:
                pass  # Just in case the actor is not alive

    def _end_junction_behavior(self, junction):
        """
        Destroys unneeded actors (those that aren't part of the route's road),
        moving the rest to other data structures and cleaning up the variables.
        If no other junctions are active, starts road mode
        """
        actor_dict = junction.actor_dict
        route_exit_keys = junction.route_exit_keys
        self._active_junctions.pop(0)

        # Prepare the road dictionary
        if not self._active_junctions:
            for wp in junction.exit_wps:
                if get_lane_key(wp) in route_exit_keys:
                    self._road_dict[get_lane_key(wp)] = Source(wp, [], active=self._route_sources_active)

        else:
            for wp in junction.exit_wps:
                if get_lane_key(wp) in route_exit_keys:
                    # TODO: entry_lane_wp isn't really this one (for cases with road changes)
                    self._active_junctions[0].entry_sources.append(
                        Source(wp, [], entry_lane_wp=wp, active=self._route_sources_active)
                    )

        # Handle the actors
        for actor in list(actor_dict):
            location = CarlaDataProvider.get_location(actor)
            if not location or self._is_location_behind_ego(location):
                self._destroy_actor(actor)
                continue

            # Don't destroy those that are on the route's road opposite lane.
            # Instead, let them move freely until they are automatically destroyed.
            self._tm.vehicle_percentage_speed_difference(actor, 0)
            if actor_dict[actor]['at_oppo_entry_lane']:
                self._opposite_actors.append(actor)
                self._tm.ignore_lights_percentage(actor, 100)
                self._tm.ignore_signs_percentage(actor, 100)
                continue

            # Save those that are on the route's road
            exit_key = actor_dict[actor]['exit_lane_key']
            if exit_key in route_exit_keys:
                if not self._active_junctions:
                    self._road_dict[exit_key].actors.append(actor)
                else:
                    entry_sources = self._active_junctions[0].entry_sources
                    for entry_source in entry_sources: # Add it to the back source
                        if exit_key == get_lane_key(entry_source.wp):
                            entry_sources.actors.append(actor)
                            break
                continue

            # Destroy the rest
            self._destroy_actor(actor)

        self._handle_crossing_scenario_interaction(junction, self._ego_wp) # TODO: Do a better scenario case
        self._handle_junction_scenario_end(junction)  # TODO: See if this can be removed (as it changes a popped element)

        if not self._active_junctions:
            self._ego_state = EGO_ROAD
            self._initialise_opposite_sources()
            self._initialise_road_checker()
            self._road_ego_key = self._get_ego_route_lane_key(self._ego_wp)

    def _search_for_next_junction(self):
        """Check if closeby to a junction. The closest one will always be the first"""
        if not self._junctions:
            return None

        ego_accum_dist = self._accum_dist[self._route_index]
        junction_accum_dist = self._accum_dist[self._junctions[0].route_entry_index]
        if junction_accum_dist - ego_accum_dist < self._junction_detection_dist:  # Junctions closeby
            return self._junctions.pop(0)

        return None

    def _initialise_connecting_lanes(self, junction):
        """
        Moves the actors currently at the exit lane of the last junction
        to entry actors of the newly created junction
        """
        if len(self._active_junctions) > 0:
            prev_junction = self._active_junctions[-1]
            route_exit_keys = prev_junction.route_exit_keys
            exit_dict = prev_junction.exit_dict
            for exit_key in route_exit_keys:
                exit_actors = exit_dict[exit_key]['actors']
                for actor in list(exit_actors):
                    self._remove_actor_info(actor)
                    self._add_actor_dict_element(junction.actor_dict, actor)
                    self._tm.vehicle_percentage_speed_difference(actor, 0)

    def _monitor_incoming_junctions(self):
        """
        Monitors when the ego approaches a junction, triggering that junction when it happens.
        This can be triggered even if there is another junction happening are they work independently
        """
        junction = self._search_for_next_junction()
        if not junction:
            return

        if self._ego_state == EGO_ROAD:
            self._switch_to_junction_mode(junction)
        self._initialise_junction_sources(junction)
        self._initialise_junction_exits(junction)

        self._initialise_connecting_lanes(junction)
        self._active_junctions.append(junction)

    def _monitor_ego_junction_exit(self):
        """
        Monitors when the ego exits the junctions, preparing the road mode when that happens
        """
        current_junction = self._active_junctions[0]
        exit_index = current_junction.route_exit_index
        if exit_index and self._route_index >= exit_index:
            self._end_junction_behavior(current_junction)

    def _add_incoming_actors(self, junction, source):
        """Checks nearby actors that will pass through the source, adding them to it"""
        source_location = source.wp.transform.location
        if not source.previous_lane_keys:
            source.previous_lane_keys = [get_lane_key(prev_wp) for prev_wp in source.wp.previous(self._reuse_dist)]
            source.previous_lane_keys.append(get_lane_key(source.wp))

        for actor in self._get_actors():
            if actor in source.actors:
                continue  # Don't use actors already part of the source

            actor_location = CarlaDataProvider.get_location(actor)
            if actor_location is None:
                continue  # No idea where the actor is, ignore it
            if source_location.distance(actor_location) > self._reuse_dist:
                continue  # Don't use actors far away

            actor_wp = self._map.get_waypoint(actor_location)
            if get_lane_key(actor_wp) not in source.previous_lane_keys:
                continue  # Don't use actors that won't pass through the source

            self._tm.vehicle_percentage_speed_difference(actor, 0)
            self._remove_actor_info(actor)
            source.actors.append(actor)

            at_oppo_entry_lane = get_lane_key(source.entry_lane_wp) in junction.opposite_entry_keys
            self._add_actor_dict_element(junction.actor_dict, actor, at_oppo_entry_lane=at_oppo_entry_lane)

            return actor

    def _update_road_sources(self, prev_ego_index):
        """
        Manages the sources that spawn actors behind the ego.
        These are always behidn the ego and will constinuously spawn actors.
        These sources also track the amount of vehicles in front of the ego,
        removing actors if the amount is too high.
        """
        if prev_ego_index != self._route_index:
            route_wp = self._route[self._route_index]
            prev_route_wp = self._route[prev_ego_index]
            added_dist = route_wp.transform.location.distance(prev_route_wp.transform.location)

            min_distance = self._road_back_vehicles * self._road_spawn_dist
            for lane_key in self._road_dict:
                source = self._road_dict[lane_key]
                if source.dist_to_ego < min_distance:
                    source.dist_to_ego += added_dist
                    continue  # Don't move the source that is too close to the ego,

                distance = source.wp.transform.location.distance(source.actors[-1].get_location())
                if distance <= self._road_spawn_dist:
                    continue  # or is it is too close to the last vehicle

                added_dist += distance - self._road_spawn_dist  # But don't let it stagger behind
                new_source_wps = source.wp.next(added_dist)
                if not new_source_wps:
                    continue
                source.wp = new_source_wps[0]

        for lane_key in self._road_dict:
            source = self._road_dict[lane_key]
            if self.debug:
                draw_point(self._world, source.wp.transform.location, DEBUG_SMALL, DEBUG_ROAD, False)
                draw_string(self._world, source.wp.transform.location, str(len(source.actors)), DEBUG_ROAD, False)

            # Ensure not too many actors are in front of the ego
            front_veh = 0
            for actor in source.actors:
                location = CarlaDataProvider.get_location(actor)
                if not location:
                    continue
                if not self._is_location_behind_ego(location):
                    front_veh += 1
            if front_veh > self._road_front_vehicles:
                self._destroy_actor(source.actors[0])  # This is always the front most vehicle

            if not source.active:
                continue
            if len(source.actors) >= self._road_back_vehicles + self._road_front_vehicles:
                continue

            if len(source.actors) == 0:
                location = self._ego_wp.transform.location
            else:
                location = CarlaDataProvider.get_location(source.actors[-1])
                if not location:
                    continue

            distance = location.distance(source.wp.transform.location)

            # Spawn a new actor if the last one is far enough
            if distance > self._road_spawn_dist:
                actor = self._spawn_source_actor(source, ego_dist=self._road_vehicle_dist)
                if actor is None:
                    continue

                self._tm.distance_to_leading_vehicle(actor, self._road_vehicle_dist)
                source.actors.append(actor)

    ################################
    ## Behavior related functions ##
    ################################

    def _initialise_road_behavior(self, road_wps):
        """
        Initialises the road behavior, consisting on several vehicle in front of the ego,
        and several on the back and are only spawned outside junctions.
        If there aren't enough actors behind, road sources will be created that will do so later on
        """
        # Vehicles in front
        for wp in road_wps:
            spawn_wps = []

            # Front spawn points
            next_wp = wp
            for _ in range(self._road_front_vehicles):
                next_wps = next_wp.next(self._road_spawn_dist)
                if len(next_wps) != 1 or self._is_junction(next_wps[0]):
                    break  # Stop when there's no next or found a junction
                next_wp = next_wps[0]
                spawn_wps.insert(0, next_wp)

            # Back spawn points
            source_dist = 0
            prev_wp = wp
            for _ in range(self._road_back_vehicles):
                prev_wps = prev_wp.previous(self._road_spawn_dist)
                if len(prev_wps) != 1 or self._is_junction(prev_wps[0]):
                    break  # Stop when there's no next or found a junction
                prev_wp = prev_wps[0]
                spawn_wps.append(prev_wp)
                source_dist += self._road_spawn_dist

            # Spawn actors
            actors = self._spawn_actors(spawn_wps)
            for actor in actors:
                self._tm.distance_to_leading_vehicle(actor, self._road_vehicle_dist)

            self._road_dict[get_lane_key(wp)] = Source(
                prev_wp, actors, dist_to_ego=source_dist, active=self._route_sources_active
            )

    def _initialise_opposite_sources(self):
        """
        All opposite lanes have actor sources that will continually create vehicles,
        creating the sensation of permanent traffic. The actor spawning will be done later on
        (_update_opposite_sources). These sources are at a (somewhat) fixed distance
        from the ego, but they never entering junctions. 
        """
        self._opposite_route_index = None
        if not self._junctions:
            next_junction_index = self._route_length - 1
        else:
            next_junction_index = self._junctions[0].route_entry_index

        ego_accum_dist = self._accum_dist[self._route_index]
        for i in range(self._route_index, next_junction_index):
            if self._accum_dist[i] - ego_accum_dist > self._opposite_sources_dist:
                self._opposite_route_index = i
                break
        if not self._opposite_route_index:
            # Junction is closer than the opposite source distance
            self._opposite_route_index = next_junction_index

        oppo_wp = self._route[self._opposite_route_index]

        for wp in get_opposite_dir_lanes(oppo_wp):
            self._opposite_sources.append(Source(wp, []))

    def _initialise_road_checker(self):
        """
        Gets the waypoints in front of the ego to continuously check if the road changes
        """
        self._road_checker_index = None

        if not self._junctions:
            upper_limit = self._route_length - 1
        else:
            upper_limit = self._junctions[0].route_entry_index

        ego_accum_dist = self._accum_dist[self._route_index]
        for i in range(self._route_index, upper_limit):
            if self._accum_dist[i] - ego_accum_dist > self._max_radius:
                self._road_checker_index = i
                break
        if not self._road_checker_index:
            self._road_checker_index = upper_limit

    def _initialise_junction_sources(self, junction):
        """
        Initializes the actor sources to ensure the junction is always populated. They are
        placed at certain distance from the junction, but are stopped if another junction is found,
        to ensure the spawned actors always move towards the activated one.
        """
        remove_entries = junction.scenario_info['remove_entries']
        direction = junction.scenario_info['direction']
        entry_lanes = [] if not direction else junction.entry_directions[direction]

        for wp in junction.entry_wps:
            entry_lane_key = get_lane_key(wp)
            if entry_lane_key in junction.route_entry_keys:
                continue  # Ignore the road from which the route enters

            # TODO: Use the source.active to do this
            if remove_entries and entry_lane_key in entry_lanes:
                continue  # Ignore entries that are part of active junction scenarios

            moved_dist = 0
            prev_wp = wp
            while moved_dist < self._junction_sources_dist:
                prev_wps = prev_wp.previous(5)
                if len(prev_wps) == 0 or self._is_junction(prev_wps[0]):
                    break
                prev_wp = prev_wps[0]
                moved_dist += 5

            junction.entry_sources.append(Source(prev_wp, [], entry_lane_wp=wp))

    def _initialise_junction_exits(self, junction):
        """
        Computes and stores the max capacity of the exit. Prepares the behavior of the next road
        by creating actors at the route exit, and the sources that'll create actors behind the ego
        """
        exit_wps = junction.exit_wps
        route_exit_keys = junction.route_exit_keys

        remove_exits = junction.scenario_info['remove_exits']
        direction = junction.scenario_info['direction']
        exit_lanes = [] if not direction else junction.exit_directions[direction]

        for wp in exit_wps:
            max_actors = 0
            max_distance = junction.exit_dict[get_lane_key(wp)]['max_distance']
            exiting_wps = []

            next_wp = wp
    
            # Move the initial distance
            if max_distance > 0:
                next_wps = next_wp.next(max_distance)
                if len(next_wps) > 0:
                    next_wp = next_wps[0]

            for i in range(max(self._road_front_vehicles, 1)):

                # Get the moving distance (first jump is higher to allow space for another vehicle)
                if i == 0:
                    move_dist = 2 * self._junction_spawn_dist
                else:
                    move_dist = self._junction_spawn_dist

                # And move such distance
                next_wps = next_wp.next(move_dist)
                if len(next_wps) == 0:
                    break  # Stop when there's no next
                next_wp = next_wps[0]
                if max_actors > 0 and self._is_junction(next_wp):
                    break  # Stop when a junction is found

                max_actors += 1
                max_distance += move_dist
                exiting_wps.insert(0, next_wp)

            junction.exit_dict[get_lane_key(wp)] = {
                'actors': [], 'max_actors': max_actors, 'ref_wp': wp, 'max_distance': max_distance
            }

            exit_lane_key = get_lane_key(wp)
            if remove_exits and exit_lane_key in exit_lanes:
                continue  # The direction is prohibited as a junction scenario is active

            if exit_lane_key in route_exit_keys:
                actors = self._spawn_actors(exiting_wps)
                for actor in actors:
                    self._tm.distance_to_leading_vehicle(actor, self._junction_vehicle_dist)
                    self._add_actor_dict_element(junction.actor_dict, actor, exit_lane_key=exit_lane_key)
                junction.exit_dict[exit_lane_key]['actors'] = actors

    def _update_junction_sources(self):
        """Checks the actor sources to see if new actors have to be created"""
        for junction in self._active_junctions:
            remove_entries = junction.scenario_info['remove_entries']
            direction = junction.scenario_info['direction']
            entry_keys = [] if not direction else junction.entry_directions[direction]

            actor_dict = junction.actor_dict
            for source in junction.entry_sources:
                if self.debug:
                    draw_point(self._world, source.wp.transform.location, DEBUG_SMALL, DEBUG_JUNCTION, False)
                    draw_string(self._world, source.wp.transform.location, str(len(source.actors)), DEBUG_JUNCTION, False)

                entry_lane_key = get_lane_key(source.entry_lane_wp)
                at_oppo_entry_lane = entry_lane_key in junction.opposite_entry_keys

                if not source.active:
                    continue
                # The direction is prohibited
                if remove_entries and entry_lane_key in entry_keys:
                    continue

                self._add_incoming_actors(junction, source)

                # Cap the amount of alive actors
                if len(source.actors) >= self._junction_sources_max_actors:
                    continue

                # Calculate distance to the last created actor
                if len(source.actors) == 0:
                    actor_location = CarlaDataProvider.get_location(self._ego_actor)
                else:
                    actor_location = CarlaDataProvider.get_location(source.actors[-1])

                if not actor_location:
                    continue
                distance = actor_location.distance(source.wp.transform.location)

                # Spawn a new actor if the last one is far enough
                if distance > self._junction_spawn_dist:
                    actor = self._spawn_source_actor(source)
                    if not actor:
                        continue
                    if junction.stop_entries:
                        self._tm.vehicle_percentage_speed_difference(actor, 100)
                    self._tm.distance_to_leading_vehicle(actor, self._junction_vehicle_dist)
                    self._add_actor_dict_element(actor_dict, actor, at_oppo_entry_lane=at_oppo_entry_lane)
                    source.actors.append(actor)

    def _found_a_road_change(self, old_index, new_index, ignore_false_junctions=True):
        """Checks if the new route waypoint is part of a new road (excluding fake junctions)"""
        if new_index == old_index:
            return False

        new_wp = self._route[new_index]
        old_wp = self._route[old_index]
        if get_road_key(new_wp) == get_road_key(old_wp):
            return False

        if ignore_false_junctions:
            new_wp_junction = new_wp.get_junction()
            if new_wp_junction and new_wp_junction.id in self._fake_junction_ids:
                return False

        return True

    def _move_road_checker(self, prev_index):
        """
        Continually check the road in front to see if it has changed its topology.
        If so and the number of lanes have reduced, remove the actor of the lane that merges into others
        """
        if self.debug:
            checker_wp = self._route[self._road_checker_index]
            draw_point(self._world, checker_wp.transform.location, DEBUG_SMALL, DEBUG_ROAD, False)

        if prev_index == self._route_index:
            return

        # Get the new route tracking wp
        checker_index = None
        last_index = self._junctions[0].route_entry_index if self._junctions else self._route_length - 1
        current_accum_dist = self._accum_dist[self._route_index]
        for i in range(self._road_checker_index, last_index):
            accum_dist = self._accum_dist[i]
            if accum_dist - current_accum_dist >= self._max_radius:
                checker_index = i
                break
        if not checker_index:
            checker_index = last_index

        if self._found_a_road_change(self._road_checker_index, checker_index):
            new_wps = get_same_dir_lanes(self._route[checker_index])
            old_wps = get_same_dir_lanes(self._route[self._road_checker_index])

            if len(new_wps) >= len(old_wps):
                pass
            else:
                new_accum_dist = self._accum_dist[checker_index]
                prev_accum_dist = self._accum_dist[self._road_checker_index]
                route_move_dist = new_accum_dist - prev_accum_dist
                unmapped_lane_keys = []

                for old_wp in list(old_wps):
                    location = old_wp.transform.location
                    mapped_wp = None
                    for new_wp in new_wps:
                        if location.distance(new_wp.transform.location) < 1.1 * route_move_dist:
                            mapped_wp = new_wp
                            break

                    if not mapped_wp:
                        unmapped_lane_keys.append(get_lane_key(old_wp))

                for lane in self._road_dict:
                    if lane in unmapped_lane_keys:
                        for actor in list(self._road_dict[lane].actors):
                            self._destroy_actor(actor)

        self._road_checker_index = checker_index

    def _move_opposite_sources(self, prev_index):
        """
        Moves the sources of the opposite direction back. Additionally, tracks a point a certain distance
        in front of the ego to see if the road topology has to be recalculated
        """
        if self.debug:
            for source in self._opposite_sources:
                draw_point(self._world, source.wp.transform.location, DEBUG_SMALL, DEBUG_OPPOSITE, False)
                draw_string(self._world, source.wp.transform.location, str(len(source.actors)), DEBUG_OPPOSITE, False)
            route_wp = self._route[self._opposite_route_index]
            draw_point(self._world, route_wp.transform.location, DEBUG_SMALL, DEBUG_OPPOSITE, False)

        if prev_index == self._route_index:
            return

        # Get the new route tracking wp
        oppo_route_index = None
        last_index = self._junctions[0].route_entry_index if self._junctions else self._route_length - 1
        current_accum_dist = self._accum_dist[self._route_index]
        for i in range(self._opposite_route_index, last_index):
            accum_dist = self._accum_dist[i]
            if accum_dist - current_accum_dist >= self._opposite_sources_dist:
                oppo_route_index = i
                break
        if not oppo_route_index:
            oppo_route_index = last_index

        if self._found_a_road_change(self._opposite_route_index, oppo_route_index):
            # Recheck the left lanes as the topology might have changed
            new_opposite_sources = []
            new_opposite_wps = get_opposite_dir_lanes(self._route[oppo_route_index])

            # Map the old sources to the new wps, and add new ones / remove uneeded ones
            new_accum_dist = self._accum_dist[oppo_route_index]
            prev_accum_dist = self._accum_dist[self._opposite_route_index]
            route_move_dist = new_accum_dist - prev_accum_dist
            for wp in new_opposite_wps:
                location = wp.transform.location
                new_source = None
                for source in self._opposite_sources:
                    if location.distance(source.wp.transform.location) < 1.1 * route_move_dist:
                        new_source = source
                        break

                if new_source:
                    new_source.wp = wp
                    new_opposite_sources.append(source)
                    self._opposite_sources.remove(source)
                else:
                    new_opposite_sources.append(Source(wp, []))

            self._opposite_sources = new_opposite_sources
        else:
            prev_accum_dist = self._accum_dist[prev_index]
            current_accum_dist = self._accum_dist[self._route_index]
            move_dist = current_accum_dist - prev_accum_dist
            if move_dist <= 0:
                return

            for source in self._opposite_sources:
                wp = source.wp
                if not self._is_junction(wp):
                    prev_wps = wp.previous(move_dist)
                    if len(prev_wps) == 0:
                        continue
                    prev_wp = prev_wps[0]
                    source.wp = prev_wp

        self._opposite_route_index = oppo_route_index

    def _update_opposite_sources(self):
        """Checks the opposite actor sources to see if new actors have to be created"""
        for source in self._opposite_sources:
            # Cap the amount of alive actors
            if len(source.actors) >= self._opposite_sources_max_actors:
                continue

            # Calculate distance to the last created actor
            if len(source.actors) == 0:
                distance = self._opposite_spawn_dist + 1
            else:
                actor_location = CarlaDataProvider.get_location(source.actors[-1])
                if not actor_location:
                    continue
                distance = source.wp.transform.location.distance(actor_location)

            # Spawn a new actor if the last one is far enough
            if distance > self._opposite_spawn_dist:
                actor = self._spawn_source_actor(source)
                if actor is None:
                    continue

                self._tm.distance_to_leading_vehicle(actor, self._opposite_vehicle_dist)
                self._opposite_actors.append(actor)
                source.actors.append(actor)

    def _update_parameters(self):
        """
        Changes those parameters that have dynamic behaviors and / or that can be changed by external source.
        This is done using py_trees' Blackboard variables and all behaviors should be at `background_manager.py`.
        The blackboard variable is reset to None to avoid changing them back again next time.
        """

        # Road behavior
        road_behavior_data = py_trees.blackboard.Blackboard().get('BA_ChangeRoadBehavior')
        if road_behavior_data is not None:
            num_front_vehicles, num_back_vehicles, vehicle_dist, spawn_dist, sources_active = road_behavior_data
            if num_front_vehicles is not None:
                self._road_front_vehicles = num_front_vehicles
            if num_back_vehicles is not None:
                self._road_back_vehicles = num_back_vehicles
            if vehicle_dist is not None:
                self._road_vehicle_dist = vehicle_dist
            if spawn_dist is not None:
                self._road_spawn_dist = spawn_dist
            if sources_active is not None:
                for lane_key in self._road_dict:
                    self._road_dict[lane_key].active = sources_active

            self._get_road_radius()
            py_trees.blackboard.Blackboard().set('BA_ChangeRoadBehavior', None, True)

        # Extend the space of a specific exit lane
        road_exit_data = py_trees.blackboard.Blackboard().get('BA_ExtentExitRoadSpace')
        if road_exit_data is not None:
            space, direction = road_exit_data
            self._extent_road_exit_space(space, direction)
            py_trees.blackboard.Blackboard().set("BA_ExtentExitRoadSpace", None, True)

        # Switch route sources
        switch_sources_data = py_trees.blackboard.Blackboard().get('BA_SwitchRouteSources')
        if switch_sources_data is not None:
            self._switch_route_sources(switch_sources_data)
            py_trees.blackboard.Blackboard().set("BA_SwitchRouteSources", None, True)

        # Opposite behavior
        opposite_behavior_data = py_trees.blackboard.Blackboard().get('BA_ChangeOppositeBehavior')
        if opposite_behavior_data is not None:
            source_dist, vehicle_dist, spawn_dist, max_actors = road_behavior_data
            if source_dist is not None:
                if source_dist < self._junction_sources_dist:
                    print('WARNING: Opposite sources distance is lower than the junction ones. Ignoring it')
                else:
                    self._opposite_sources_dist = source_dist
            if vehicle_dist is not None:
                self._opposite_vehicle_dist = vehicle_dist
            if spawn_dist is not None:
                self._opposite_spawn_dist = spawn_dist
            if max_actors is not None:
                self._opposite_sources_max_actors = max_actors
            py_trees.blackboard.Blackboard().set('BA_ChangeOppositeBehavior', None, True)

        # Junction behavior
        junction_behavior_data = py_trees.blackboard.Blackboard().get('BA_ChangeJunctionBehavior')
        if junction_behavior_data is not None:
            source_dist, vehicle_dist, spawn_dist, max_actors = road_behavior_data
            if source_dist is not None:
                if source_dist > self._opposite_sources_dist:
                    print('WARNING: Junction sources distance is higher than the opposite ones. Ignoring it')
                else:
                    self._junction_sources_dist = source_dist
            if vehicle_dist is not None:
                self._junction_vehicle_dist = vehicle_dist
            if spawn_dist is not None:
                self._junction_spawn_dist = spawn_dist
            if max_actors is not None:
                self._junction_sources_max_actors = max_actors
            py_trees.blackboard.Blackboard().set('BA_ChangeJunctionBehavior', None, True)

        # Stop front vehicles
        stop_data = py_trees.blackboard.Blackboard().get('BA_StopFrontVehicles')
        if stop_data is not None:
            self._stop_road_front_vehicles()
            py_trees.blackboard.Blackboard().set('BA_StopFrontVehicles', None, True)

        # Start front vehicles
        start_data = py_trees.blackboard.Blackboard().get('BA_StartFrontVehicles')
        if start_data is not None:
            self._start_road_front_vehicles()
            py_trees.blackboard.Blackboard().set("BA_StartFrontVehicles", None, True)

        # Handles crossing actor scenarios. This currently only works for Scenario4
        crossing_dist = py_trees.blackboard.Blackboard().get('BA_HandleCrossingActor')
        if crossing_dist is not None:
            self._is_crossing_scenario_active = True
            self._crossing_dist = crossing_dist
            py_trees.blackboard.Blackboard().set('BA_HandleCrossingActor', None, True)

        # Handles junction scenarios (scenarios 7 to 10)
        junction_scenario_data = py_trees.blackboard.Blackboard().get('BA_JunctionScenario')
        if junction_scenario_data is not None:
            direction, remove_entries, remove_exits, remove_middle = junction_scenario_data
            self._initialise_junction_scenario(direction, remove_entries, remove_exits, remove_middle)
            py_trees.blackboard.Blackboard().set('BA_JunctionScenario', None, True)

        # Handles road accident scenario (Accident and Construction)
        handle_start_accident_data = py_trees.blackboard.Blackboard().get('BA_HandleStartAccidentScenario')
        if handle_start_accident_data is not None:
            accident_wp, distance = handle_start_accident_data
            self._handle_lanechange_scenario(accident_wp, distance)
            py_trees.blackboard.Blackboard().set('BA_HandleStartAccidentScenario', None, True)

        # Handles road accident scenario (Accident and Construction)
        handle_end_accident_data = py_trees.blackboard.Blackboard().get('BA_HandleEndAccidentScenario')
        if handle_end_accident_data is not None:
            self._road_extra_front_actors = 0
            py_trees.blackboard.Blackboard().set('BA_HandleEndAccidentScenario', None, True)

        # Stops non route entries
        stop_entry_data = py_trees.blackboard.Blackboard().get('BA_StopEntries')
        if stop_entry_data is not None:
            self._stop_non_route_entries()
            py_trees.blackboard.Blackboard().set('BA_StopEntries', None, True)

        self._compute_parameters()

    def _compute_parameters(self):
        """Computes the parameters that are dependent on the speed of the ego. """
        ego_speed = CarlaDataProvider.get_velocity(self._ego_actor)

        # As the vehicles don't move if the agent doesn't, some agents might get blocked forever.
        # Partially avoid this by adding an extra distance to the radius when the vehicle is stopped
        # in the middle of the road and unaffected by any object such as traffic lights or stops.
        if ego_speed == 0 \
                and len(self._stopped_road_actors) == 0 \
                and not self._ego_actor.is_at_traffic_light() \
                and len(self._active_junctions) <= 0:
            self._extra_radius = min(self._extra_radius + self._extra_radius_increase_ratio, self._max_extra_radius)

        # At all cases, reduce it if the agent is moving
        if ego_speed > 0 and self._extra_radius > 0:
            self._extra_radius = max(self._extra_radius - self._extra_radius_increase_ratio, 0)

        self._min_radius = self._base_min_radius + self._radius_increase_ratio * ego_speed + self._extra_radius
        self._max_radius = self._base_max_radius + self._radius_increase_ratio * ego_speed + self._extra_radius
        self._junction_detection_dist = self._max_radius

    def _stop_road_front_vehicles(self):
        """
        Manages the break scenario, where all road vehicles in front of the ego suddenly stop,
        wait for a bit, and start moving again. This will never trigger unless done so from outside.
        """
        for lane in self._road_dict:
            for actor in self._road_dict[lane].actors:
                location = CarlaDataProvider.get_location(actor)
                if location and not self._is_location_behind_ego(location):
                    self._stopped_road_actors.append(actor)
                    self._tm.vehicle_percentage_speed_difference(actor, 100)
                    lights = actor.get_light_state()
                    lights |= carla.VehicleLightState.Brake
                    actor.set_light_state(carla.VehicleLightState(lights))

    def _start_road_front_vehicles(self):
        """
        Manages the break scenario, where all road vehicles in front of the ego suddenly stop,
        wait for a bit, and start moving again. This will never trigger unless done so from outside.
        """
        for actor in self._stopped_road_actors:
            self._tm.vehicle_percentage_speed_difference(actor, 0)
            lights = actor.get_light_state()
            lights &= ~carla.VehicleLightState.Brake
            actor.set_light_state(carla.VehicleLightState(lights))
        self._stopped_road_actors = []

    def _extent_road_exit_space(self, space, direction):
        """Increases the space left by the exit vehicles at a specific road"""
        if len(self._active_junctions) > 0:
            junction = self._active_junctions[0]
        elif len(self._junctions) > 0:
            junction = self._junctions[0]
        else:
            return

        direction_lane_keys = junction.exit_directions[direction]
        exit_dict = junction.exit_dict
        for exit_lane_key in exit_dict:
            if exit_lane_key in direction_lane_keys:
                exit_dict[exit_lane_key]['max_distance'] += space

                # For all the actors present, teleport them a bit to the front and activate them
                for actor in exit_dict[exit_lane_key]['actors']:
                    location = CarlaDataProvider.get_location(actor)
                    if not location:
                        continue

                    actor_wp = self._map.get_waypoint(location)
                    new_actor_wps = actor_wp.next(space)
                    if len(new_actor_wps) > 0:
                        new_transform = new_actor_wps[0].transform
                        new_transform.location.z += 0.2
                        actor.set_transform(new_transform)

                    if junction.actor_dict[actor]['state'] == JUNCTION_INACTIVE:
                        self._tm.vehicle_percentage_speed_difference(actor, 0)
                        junction.actor_dict[actor]['state'] = JUNCTION_EXIT


    def _stop_non_route_entries(self):
        if len(self._active_junctions) > 0:
            entry_sources = self._active_junctions[0].entry_sources
            for source in entry_sources:
                for actor in source.actors:
                    self._tm.vehicle_percentage_speed_difference(actor, 100)
        elif len(self._junctions) > 0:
            self._junctions[0].stop_entries = True

    def _handle_lanechange_scenario(self, accident_wp, distance):
        """
        Handles the scenario in which the BA has to change lane.
        """
        ego_wp = self._route[self._route_index]
        lane_change_actors = self._road_dict[get_lane_key(ego_wp)].actors
        for actor in lane_change_actors:
            location = CarlaDataProvider.get_location(actor)
            if not self._is_location_behind_ego(location):
                lanechange_wp = accident_wp.get_left_lane().next(distance/4)[0]
                end_lanechange_wp = lanechange_wp.next(distance/2)[0]
                vehicle_path = [lanechange_wp.transform.location,
                                    end_lanechange_wp.transform.location,
                                    end_lanechange_wp.get_right_lane().next(distance/4)[0].transform.location]
                self._tm.set_path(actor, vehicle_path)
                ## maybe check here to activate lane changing lights
                self._road_extra_front_actors += 1
            else:
                self._tm.vehicle_percentage_speed_difference(actor, 100)

    def _switch_route_sources(self, enabled):
        """
        Disables all sources that are part of the ego's route
        """
        self._route_sources_active = enabled
        for lane in self._road_dict:
            self._road_dict[lane].active = enabled

        for junction in self._active_junctions:
            for source in junction.entry_sources:
                if get_lane_key(source.entry_lane_wp) in junction.route_entry_keys:
                    source.active = enabled

    #############################
    ##     Actor functions     ##
    #############################

    def _spawn_actors(self, spawn_wps):
        """Spawns several actors in batch"""
        spawn_transforms = []
        for wp in spawn_wps:
            spawn_transforms.append(
                carla.Transform(wp.transform.location + carla.Location(z=self._spawn_vertical_shift),
                                wp.transform.rotation)
            )

        actors = CarlaDataProvider.request_new_batch_actors(
            'vehicle.*', len(spawn_transforms), spawn_transforms, True, False, 'background',
            filter_type="car", tick=False)

        if not actors:
            return actors

        for actor in actors:
            self._tm.auto_lane_change(actor, False)

        if self._night_mode:
            for actor in actors:
                actor.set_light_state(carla.VehicleLightState(
                    carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam))

        return actors

    def _spawn_source_actor(self, source, ego_dist=0):
        """Given a source, spawns an actor at that source"""
        ego_location = CarlaDataProvider.get_location(self._ego_actor)
        source_transform = source.wp.transform
        if ego_location.distance(source_transform.location) < ego_dist:
            return None

        new_transform = carla.Transform(
            source_transform.location + carla.Location(z=self._spawn_vertical_shift),
            source_transform.rotation
        )
        actor = CarlaDataProvider.request_new_actor(
            'vehicle.*', new_transform, rolename='background',
            autopilot=True, random_location=False, filter_type="car", tick=False)

        if not actor:
            return actor

        self._tm.auto_lane_change(actor, False)
        if self._night_mode:
            actor.set_light_state(carla.VehicleLightState(
                carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam))

        return actor

    def _is_location_behind_ego(self, location):
        """Checks if an actor is behind the ego. Uses the route transform"""
        ego_transform = self._route[self._route_index].transform
        ego_heading = ego_transform.get_forward_vector()
        ego_actor_vec = location - ego_transform.location
        if ego_heading.x * ego_actor_vec.x + ego_heading.y * ego_actor_vec.y < - 0.17:  # 100º
            return True
        return False

    def _get_ego_route_lane_key(self, route_wp):
        """
        Gets the route lane key of the ego. This corresponds to the same lane if the ego is driving normally,
        but if is is going in opposite direction, the route's leftmost one is chosen instead
        """
        location = CarlaDataProvider.get_location(self._ego_actor)
        ego_true_wp = self._map.get_waypoint(location)
        if get_road_key(ego_true_wp) != get_road_key(route_wp):
            # Just return the default value as two different roads are being compared.
            # This might happen for when moving to a new road and should be fixed next frame
            return get_lane_key(route_wp)

        yaw_diff = (ego_true_wp.transform.rotation.yaw - route_wp.transform.rotation.yaw) % 360
        if yaw_diff < 90 or yaw_diff > 270:
            return get_lane_key(ego_true_wp)
        else:
            # Get the first lane of the opposite direction
            leftmost_wp = route_wp
            while True:
                possible_left_wp = leftmost_wp.get_left_lane()
                if possible_left_wp is None or possible_left_wp.lane_id * leftmost_wp.lane_id < 0:
                    break
                leftmost_wp = possible_left_wp
            return get_lane_key(leftmost_wp)

    def _update_road_actors(self):
        """
        Dynamically controls the actor speed in front of the ego.
        Not applied to those behind it so that they can catch up it
        """
        # Updates their speed
        route_wp = self._route[self._route_index]
        scenario_actors = self._crossing_scenario_actors + self._stopped_road_actors
        for lane in self._road_dict:
            for i, actor in enumerate(self._road_dict[lane].actors):
                location = CarlaDataProvider.get_location(actor)
                if not location:
                    continue
                if self.debug:
                    string = 'R_'
                    string += 'B' if self._is_location_behind_ego(location) else 'F'
                    string += '(' + str(i) + ')'
                    draw_string(self._world, location, string, DEBUG_ROAD, False)
                if actor in scenario_actors or self._is_location_behind_ego(location):
                    continue

                distance = location.distance(route_wp.transform.location)
                speed_red = (distance - self._min_radius) / (self._max_radius - self._min_radius) * 100
                speed_red = np.clip(speed_red, 0, 100)
                self._tm.vehicle_percentage_speed_difference(actor, speed_red)

        if not self._road_ego_key in self._road_dict:
            return

        front_actors = 0
        for actor in self._road_dict[self._road_ego_key].actors:
            if not self._is_location_behind_ego(actor.get_location()):
                front_actors += 1
        self._road_num_front_vehicles = front_actors

        self._get_road_radius()
        self._compute_parameters()

    def _monitor_road_changes(self, prev_route_index):
        """
        Checks if the ego changes road, remapping the route keys, and creating new sources (if needed)
        """
        route_wp = self._route[ self._route_index]
        prev_route_wp = self._route[prev_route_index]
        check_dist = 1.1 * route_wp.transform.location.distance(prev_route_wp.transform.location)
        if prev_route_index !=  self._route_index:
            road_change = self._found_a_road_change(prev_route_index, self._route_index, ignore_false_junctions=False)
            if not self._is_junction(prev_route_wp) and road_change:
                # Get all the wps of the new road
                if not route_wp.is_junction:
                    new_wps = get_same_dir_lanes(route_wp)
                else:  # Entering a false junction
                    new_wps = []
                    for enter_wp, _ in route_wp.get_junction().get_waypoints(carla.LaneType.Driving):
                        if get_road_key(enter_wp) == get_road_key(route_wp):
                            new_wps.append(enter_wp)

                # Get all the wps of the old road
                if not prev_route_wp.is_junction:
                    old_wps = get_same_dir_lanes(prev_route_wp)
                else:  # Exitting a false junction
                    old_wps = []
                    for _, exit_wp in prev_route_wp.get_junction().get_waypoints(carla.LaneType.Driving):
                        if get_road_key(exit_wp) == get_road_key(prev_route_wp):
                            old_wps.append(exit_wp)

                # Map the new lanes to the old ones
                mapped_keys = {}
                unmapped_wps = new_wps
                for old_wp in list(old_wps):
                    location = old_wp.transform.location
                    mapped_wp = None
                    for new_wp in unmapped_wps:
                        if location.distance(new_wp.transform.location) < check_dist:
                            mapped_wp = new_wp
                            break

                    if mapped_wp:
                        unmapped_wps.remove(mapped_wp)
                        old_wps.remove(old_wp)
                        mapped_keys[get_lane_key(old_wp)] = get_lane_key(mapped_wp)

                # Remake the road back actors dictionary
                new_road_dict = {}
                for lane_key in self._road_dict:
                    if lane_key not in mapped_keys:
                        continue  # A lane ended at that road
                    new_lane_key = mapped_keys[lane_key]
                    new_road_dict[new_lane_key] = self._road_dict[lane_key]

                self._road_dict = new_road_dict

                # New lanes, add new sources
                for unmapped_wp in unmapped_wps:
                    source_wps = unmapped_wp.next(self._road_new_sources_dist)
                    if len(source_wps) != 1:
                        continue
                    new_lane_wp = source_wps[0]
                    self._road_dict[get_lane_key(new_lane_wp)] = Source(new_lane_wp, [])

                if not self._road_ego_key in mapped_keys:
                    # Return the default. This might happen when the route lane ends and should be fixed next frame
                    self._road_ego_key = get_lane_key(route_wp)
                else:
                    self._road_ego_key = mapped_keys[self._road_ego_key]
            else:
                self._road_ego_key = self._get_ego_route_lane_key(route_wp)

    def _update_junction_actors(self):
        """
        Handles an actor depending on their previous state. Actors entering the junction have its exit
        monitored through their waypoint. When they exit, they are either moved to a connecting junction,
        or added to the exit dictionary. Actors that exited the junction will stop after a certain distance
        """
        if len(self._active_junctions) == 0:
            return

        max_index = len(self._active_junctions) - 1
        for i, junction in enumerate(self._active_junctions):
            if self.debug:
                route_keys = junction.route_entry_keys + junction.route_exit_keys
                route_oppo_keys = junction.opposite_entry_keys + junction.opposite_exit_keys
                for wp in junction.entry_wps + junction.exit_wps:
                    if get_lane_key(wp) in route_keys:
                        draw_point(self._world, wp.transform.location, DEBUG_MEDIUM, DEBUG_ROAD, False)
                    elif get_lane_key(wp) in route_oppo_keys:
                        draw_point(self._world, wp.transform.location, DEBUG_MEDIUM, DEBUG_OPPOSITE, False)
                    else:
                        draw_point(self._world, wp.transform.location, DEBUG_MEDIUM, DEBUG_JUNCTION, False)

            actor_dict = junction.actor_dict
            exit_dict = junction.exit_dict
            remove_middle = junction.scenario_info['remove_middle']
            for j, actor in enumerate(list(actor_dict)):
                if actor not in actor_dict:
                    continue  # Actor was removed during the loop
                location = CarlaDataProvider.get_location(actor)
                if not location:
                    continue

                state, exit_lane_key, _ = actor_dict[actor].values()
                if self.debug:
                    string = 'J' + str(i+1) + '_' + state[:2] + '(' + str(j) + ')'
                    draw_string(self._world, location, string, DEBUG_JUNCTION, False)

                # Monitor its entry
                if state == JUNCTION_ENTRY:
                    actor_wp = self._map.get_waypoint(location)
                    if self._is_junction(actor_wp) and junction.contains(actor_wp.get_junction()):
                        if remove_middle:
                            self._destroy_actor(actor)  # Don't clutter the junction if a junction scenario is active
                            continue
                        actor_dict[actor]['state'] = JUNCTION_MIDDLE

                # Monitor its exit and destroy an actor if needed
                elif state == JUNCTION_MIDDLE:
                    actor_wp = self._map.get_waypoint(location)
                    actor_lane_key = get_lane_key(actor_wp)
                    if not self._is_junction(actor_wp) and actor_lane_key in exit_dict:
                        if i < max_index and actor_lane_key in junction.route_exit_keys:
                            # Exited through a connecting lane in the route direction.
                            self._remove_actor_info(actor)
                            other_junction = self._active_junctions[i+1]
                            self._add_actor_dict_element(other_junction.actor_dict, actor)

                        elif i > 0 and actor_lane_key in junction.opposite_exit_keys:
                            # Exited through a connecting lane in the opposite direction.
                            # THIS SHOULD NEVER HAPPEN, an entry source should have already added it.
                            other_junction = self._active_junctions[i-1]
                            if actor not in other_junction.actor_dict:
                                self._remove_actor_info(actor)
                                self._add_actor_dict_element(other_junction.actor_dict, actor, at_oppo_entry_lane=True)

                        else:
                            # Check the lane capacity
                            exit_dict[actor_lane_key]['ref_wp'] = actor_wp
                            actor_dict[actor]['state'] = JUNCTION_EXIT
                            actor_dict[actor]['exit_lane_key'] = actor_lane_key

                            actors = exit_dict[actor_lane_key]['actors']
                            if len(actors) > 0 and len(actors) >= exit_dict[actor_lane_key]['max_actors']:
                                self._destroy_actor(actors[-1])  # This is always the front most vehicle
                            actors.append(actor)

                # Deactivate them when far from the junction
                elif state == JUNCTION_EXIT:
                    distance = location.distance(exit_dict[exit_lane_key]['ref_wp'].transform.location)
                    if distance > exit_dict[exit_lane_key]['max_distance']:
                        self._tm.vehicle_percentage_speed_difference(actor, 100)
                        actor_dict[actor]['state'] = JUNCTION_INACTIVE

                # Wait for something to happen
                elif state == JUNCTION_INACTIVE:
                    pass

    def _update_opposite_actors(self):
        """
        Updates the opposite actors. This involves tracking their position,
        removing them if too far behind the ego.
        """
        max_dist = max(self._opposite_removal_dist, self._opposite_spawn_dist)
        for actor in list(self._opposite_actors):
            location = CarlaDataProvider.get_location(actor)
            if not location:
                continue
            if self.debug:
                draw_string(self._world, location, 'O', DEBUG_OPPOSITE, False)
            distance = location.distance(self._ego_wp.transform.location)
            if distance > max_dist and self._is_location_behind_ego(location):
                self._destroy_actor(actor)

    def _remove_actor_info(self, actor):
        """Removes all the references of the actor"""

        for lane in self._road_dict:
            if actor in self._road_dict[lane].actors:
                self._road_dict[lane].actors.remove(actor)
                break

        if actor in self._opposite_actors:
            self._opposite_actors.remove(actor)
        if actor in self._stopped_road_actors:
            self._stopped_road_actors.remove(actor)
        if actor in self._crossing_scenario_actors:
            self._crossing_scenario_actors.remove(actor)

        for opposite_source in self._opposite_sources:
            if actor in opposite_source.actors:
                opposite_source.actors.remove(actor)
                break

        for junction in self._active_junctions:
            junction.actor_dict.pop(actor, None)

            for entry_source in junction.entry_sources:
                if actor in entry_source.actors:
                    entry_source.actors.remove(actor)
                    break

            for exit_keys in junction.exit_dict:
                exit_actors = junction.exit_dict[exit_keys]['actors']
                if actor in exit_actors:
                    exit_actors.remove(actor)
                    break

    def _destroy_actor(self, actor):
        """Destroy the actor and all its references"""
        self._remove_actor_info(actor)
        try:
            actor.destroy()
        except RuntimeError:
            pass

    def _update_ego_route_location(self):
        """
        Checks the ego location to see if i has moved closer to the next route waypoint,
        updating its information. This never checks for backwards movements to avoid unnedded confusion
        """
        location = CarlaDataProvider.get_location(self._ego_actor)

        for index in range(self._route_index, min(self._route_index + self._route_buffer, self._route_length)):
            route_wp = self._route[index]

            route_wp_dir = route_wp.transform.get_forward_vector()    # Waypoint's forward vector
            veh_wp_dir = location - route_wp.transform.location       # vector waypoint - vehicle
            dot_ve_wp = veh_wp_dir.x * route_wp_dir.x + veh_wp_dir.y * route_wp_dir.y + veh_wp_dir.z * route_wp_dir.z

            if dot_ve_wp > 0:
                self._route_index = index

        self._ego_wp = self._route[self._route_index]

        if self.debug:
            string = 'EGO_' + self._ego_state[0].upper()
            debug_name = DEBUG_ROAD if self._ego_state == EGO_ROAD else DEBUG_JUNCTION
            draw_string(self._world, location, string, debug_name, False)
