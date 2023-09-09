from enum import IntEnum
from typing import Any


class map:
    map_file = ""
    routes = []
    junctions = []
    driving_rule = []

    # method
    def odr_to_route_point(self, road_id, lane_id, s, t):
        pass

    def xyz_to_route_point(self, x, y, z):
        pass

    def route_point_to_xyz(self, route_point):
        pass

    def outer_side(self):
        pass

    def inner_side(self):
        pass

    def create_route(self, routes, connect_points_by, legal_route):
        pass

    def create_route_point(self, route, s, t):
        pass

    def create_xyz_point(self, x, y, z, connect_as):
        pass

    def create_odr_point(self, road_id, lane_id, s, t):
        pass

    def create_path(self):
        pass

    def create_trajectory(self):
        pass

    def resolve_relative_path(self):
        pass

    def resolve_relative_trajectory(self):
        pass

    def get_map_file(self):
        pass

    # modifier
    def number_of_lanes(self, route, num_of_lanes, lane_type, lane_use, directionality):
        pass

    def routes_are_in_sequence(self, preceding, succeeding, road):
        pass

    def roads_follow_in_junction(
        self,
        junction,
        in_road,
        out_road,
        direction,
        clockwise_count,
        number_of_roads,
        in_lane,
        out_lane,
        junction_route,
        resulting_route,
    ):
        pass

    def routes_overlap(route1, route2, overlap_kind):
        pass

    def lane_side(lane1, side, lane2, count, lane_section):
        pass

    def compound_lane_side(lane1, side, lane2, count, route):
        pass

    def end_lane(lane):
        pass

    def start_lane(lane):
        pass

    def crossing_connects(crossing, start_lane, end_lane, start_s_coord, start_angle):
        pass

    def routes_are_opposite(route1, route2, containing_road, lateral_overlap):
        pass

    @staticmethod
    def set_map_file(file: str) -> Any:
        pass


class driving_rule(IntEnum):
    left_hand_traffic = (1,)
    right_hand_traffic = 2


class junction:
    def __init__(self) -> None:
        self.roads = []


class route:
    # children compound_route, path, route_element
    pass


class directionality(IntEnum):
    pass


class route_element(route):
    # crossing, lane, lane_section, odr_point, path, road, route_point, xyz_point
    pass


class road(route_element):
    pass


class lane_section(route_element):
    pass


class lane(route_element):
    pass


class crossing(route_element):
    pass


class lane_type(IntEnum):
    pass


class lane_use(IntEnum):
    pass


class side_left_right(IntEnum):
    pass


class lon_lat(IntEnum):
    pass


class crossing_type:
    pass


class crossing_marking(IntEnum):
    pass


class crossing_use(IntEnum):
    pass


class crossing_elevation(IntEnum):
    pass


class compound_route(route):
    pass


class compound_lane(compound_route):
    pass


class junction_direction(IntEnum):
    pass


class route_overlap_kind(IntEnum):
    pass


class lateral_overlap_kind(IntEnum):
    pass


class route_point(route_element):
    pass


class xyz_point(route_element):
    pass


class odr_point(route_element):
    pass


class connect_route_points(IntEnum):
    pass


class path(route, route_element):
    pass


class relative_path:
    # relative_path_odr, relative_path_pose_3d, relative_path_st
    pass


class relative_path_pose_3d(relative_path):
    pass


class relative_path_st(relative_path):
    pass


class relative_path_odr(relative_path):
    pass


class relative_transform(IntEnum):
    pass


class trajectory:
    pass


class relative_trajectory:
    # relative_trajectory_odr, relative_trajectory_pose_3d, relative_trajectory_st
    pass


class relative_trajectory_pose_3d(relative_trajectory):
    pass


class relative_trajectory_st(relative_trajectory):
    pass


class relative_trajectory_odr(relative_trajectory):
    pass
