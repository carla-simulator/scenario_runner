import sys
from math import sqrt

import py_trees

from srunner.osc2_stdlib import variables
from srunner.osc2_stdlib.observer import Observer
from srunner.osc2_stdlib.variables import Variable
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import InTriggerNearCollision, DriveDistance


class Event:
    ego_vehicle_location = None
    npc_location = None
    other_car = None
    distance = None

    @classmethod
    def drive_distance(cls, other_car, distance):
        # cls.other_car = str(variables.Variable.get_arg("other_car"))
        # cls.distance = float(str(variables.Variable.get_arg("distance")))
        cls.other_car = other_car
        cls.distance = distance
        actor = CarlaDataProvider.get_actor_by_name(cls.other_car)
        ret = DriveDistance(actor, cls.distance)
        return ret

    @classmethod
    def abs_distance_between_locations(cls, reference_actor, actor):
        print(reference_actor, actor)
        reference_actor = "ego_vehicle"
        actor = "npc"
        actor1 = CarlaDataProvider.get_actor_by_name(reference_actor)
        actor2 = CarlaDataProvider.get_actor_by_name(actor)
        transform_ego_x = actor1.get_transform().location.x
        transform_ego_y = actor1.get_transform().location.y
        transform_npc_x = actor2.get_transform().location.x
        transform_npc_y = actor2.get_transform().location.y
        print(transform_ego_x, transform_ego_y, transform_npc_x, transform_npc_y)
        x = abs(transform_npc_x - transform_ego_x)
        y = abs(transform_npc_y - transform_ego_y)
        print(x, y)
        dis = sqrt(pow(x, 2) + pow(y, 2))
        func_name = sys._getframe().f_code.co_name
        Variable.set_arg({func_name: dis})
        print(111111111111111, Variable.get_arg(func_name))
        return dis

    @classmethod
    def near_collision(cls, other_car, distance):
        cls.other_car = other_car
        cls.distance = float(distance)
        actor = CarlaDataProvider.get_actor_by_name(cls.other_car)
        reference_actor = CarlaDataProvider.get_actor_by_name("ego_vehicle")
        ret = InTriggerNearCollision(reference_actor, actor, cls.distance)
        return ret


class NearCollision(Observer):
    def update(self, EventListener, *args, **kwargs):
        if EventListener.handling_collisions():
            print("Too close, collision imminent!")
