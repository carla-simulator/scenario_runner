from enum import Enum

class TrafficEventType(Enum):
    """
    This enum represents different traffic events that occur during driving.
    """
    NORMAL_DRIVING = 0
    COLLISION_STATIC = 1
    COLLISION_VEHICLE = 2
    COLLISION_PEDESTRIAN = 3
    ROUTE_DEVIATION = 4
    ROUTE_COMPLETION = 5
    ROUTE_COMPLETED = 6
    TRAFFIC_LIGHT_INFRACTION = 7
    WRONG_WAY_INFRACTION = 8


class TrafficEvent(object):
    def __init__(self, type, message=None, dict=None):
        """
        Initialize object

        :param type: TrafficEventType defining the type of traffic event
        :param message: optional message to inform users of the event
        :param dict: optional dictionary with arbitrary keys and values
        """
        self._type = type
        self._message = message
        self._dict = dict

    def get_type(self):
        return self._type

    def get_message(self):
        if self._message:
            return self._message
        else:
            return ""

    def set_message(self, message):
        self._message = message

    def get_dict(self):
        return self._dict

    def set_dict(self, dict):
        self._dict = dict