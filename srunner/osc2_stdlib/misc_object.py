from enum import Enum


class AVCarSide(Enum):
    behind = 1
    ahead_of = 2


class ScenarioEvent(Enum):
    start = 1
    end = 2
    all = 3


class Position:
    def __init__(self) -> None:
        pass


class LanePosition(Position):
    def __init__(self, road_id, lane_id, offset, s) -> None:
        self.road_id = road_id
        self.lane_id = lane_id
        self.offset = offset
        self.s = s


class WorldPosition(Position):
    def __init__(self, x=0, y=0, z=0, pitch=0, yaw=0, roll=0) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll
