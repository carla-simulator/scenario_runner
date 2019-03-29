import bisect

import carla
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

class ActorWrapperVertical(object):
    def __init__(self, actor):
        self.actor = actor

    def __lt__(self, other):
        location_self = self.actor.get_location()
        location_other = other.actor.get_location()

        if location_self.y < location_other.y:
            return True
        elif location_self.y > location_other.y:
            return False
        else:
            return location_self.x < location_other.x

class ActorWrapperHorizontal(object):
    def __init__(self, actor):
        self.actor = actor

    def __lt__(self, other):
        location_self = self.actor.get_location()
        location_other = other.actor.get_location()

        if location_self.x < location_other.x:
            return True
        elif location_self.x > location_other.x:
            return False
        else:
            return location_self.y < location_other.y


class OrderedList(object):
    def __init__(self):
        self._list = []

    def insert(self, value):
        bisect.insort(self._list, value)

    def __len__(self):
        return len(self._list)

    def __getitem__(self, index):
        return self._list[index]

class CollisionManager(object):
    DISTANCE_THRESHOLD = 1.5 # meters

    def __init__(self, search_window=3):
        self._search_window = search_window
        self._list_objects_vertical = OrderedList()
        self._list_objects_horizontal = OrderedList()
        self._N = 0

    def add(self, actor):
        self._list_objects_vertical.insert(ActorWrapperVertical(actor))
        self._list_objects_horizontal.insert(ActorWrapperHorizontal(actor))
        self._N += 1

    def detect_collisions(self):
        collision_dict = {}
        collision_list = []

        for anchor_index in range(len(self._list_objects_vertical)):
            anchor_actor = self._list_objects_vertical[anchor_index]
            min_index = max(anchor_index - self._search_window, 0)
            max_index = min(anchor_index + self._search_window, self._N)

            for other_index in range(min_index, max_index):
                if other_index == anchor_index:
                    continue
                other_actor = self._list_objects_vertical[other_index]
                if self._is_collision(anchor_actor, other_actor):
                    # potential collision detected
                    key_anchor = anchor_actor.actor.id
                    key_other = other_actor.actor.id

                    if key_anchor < key_other:
                        key_id = "{}.{}".format(key_anchor, key_other)
                    else:
                        key_id = "{}.{}".format(key_other, key_anchor)

                    if key_id not in collision_dict:
                        collision_list.append((anchor_actor.actor, other_actor.actor))
                        collision_dict[key_id] = True

        for anchor_index in range(len(self._list_objects_horizontal)):
            anchor_actor = self._list_objects_horizontal[anchor_index]
            min_index = max(anchor_index - self._search_window, 0)
            max_index = min(anchor_index + self._search_window, self._N)

            for other_index in range(min_index, max_index):
                if other_index == anchor_index:
                    continue
                other_actor = self._list_objects_horizontal[other_index]
                if self._is_collision(anchor_actor, other_actor):
                    # potential collision detected
                    key_anchor = anchor_actor.actor.id
                    key_other = other_actor.actor.id

                    if key_anchor < key_other:
                        key_id = "{}.{}".format(key_anchor, key_other)
                    else:
                        key_id = "{}.{}".format(key_other, key_anchor)

                    if key_id not in collision_dict:
                        collision_list.append((anchor_actor.actor, other_actor.actor))
                        collision_dict[key_id] = True

        return collision_list

    def _is_collision(self, anchor_actor, other_actor):

        anchor_transform = anchor_actor.actor.get_transform()
        anchor_bb = anchor_actor.actor.bounding_box

        anchor_A = anchor_transform.transform(anchor_bb.location + carla.Location(-anchor_bb.extent.x,
                                                                                   -anchor_bb.extent.y))
        anchor_B = anchor_transform.transform(anchor_bb.location + carla.Location(anchor_bb.extent.x,
                                                                                  -anchor_bb.extent.y))
        anchor_C = anchor_transform.transform(anchor_bb.location + carla.Location(anchor_bb.extent.x,
                                                                                  anchor_bb.extent.y))
        anchor_D = anchor_transform.transform(anchor_bb.location + carla.Location(-anchor_bb.extent.x,
                                                                                   anchor_bb.extent.y))
        other_transform = other_actor.actor.get_transform()
        other_bb = other_actor.actor.bounding_box

        other_A = other_transform.transform(other_bb.location + carla.Location(-other_bb.extent.x,
                                                                                   -other_bb.extent.y))
        other_B = other_transform.transform(other_bb.location + carla.Location(other_bb.extent.x,
                                                                                   -other_bb.extent.y))
        other_C = other_transform.transform(other_bb.location + carla.Location(other_bb.extent.x,
                                                                               other_bb.extent.y))
        other_D = other_transform.transform(other_bb.location + carla.Location(-other_bb.extent.x,
                                                                                other_bb.extent.y))

        for anchor_point in [anchor_A, anchor_B, anchor_C, anchor_D]:
            if self.point_inside_boundingbox(anchor_point, other_A, other_B, other_D):
                return True
        for other_point in [other_A, other_B, other_C, other_D]:
            if self.point_inside_boundingbox(other_point, anchor_A, anchor_B, anchor_D):
                return True

        return False

    def point_inside_boundingbox(self, p, A, B, D):
        """
        X
        :param p:
        :param bb_center:
        :param bb_extent:
        :return:
        """

        A = carla.Vector2D(A.x, A.y)
        B = carla.Vector2D(B.x, B.y)
        D = carla.Vector2D(D.x, D.y)
        M = carla.Vector2D(p.x, p.y)

        AB = B - A
        AD = D - A
        AM = M - A
        am_ab = AM.x * AB.x + AM.y * AB.y
        ab_ab = AB.x * AB.x + AB.y * AB.y
        am_ad = AM.x * AD.x + AM.y * AD.y
        ad_ad = AD.x * AD.x + AD.y * AD.y

        return am_ab > 0 and am_ab < ab_ab and am_ad > 0 and am_ad < ad_ad