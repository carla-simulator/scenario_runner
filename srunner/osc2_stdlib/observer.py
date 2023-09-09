from abc import ABCMeta, abstractmethod


class EventListener:
    def __init__(self):
        self.__observers = []
        self.flag = False

    def handling_collisions(self):
        return self.flag

    def near_collision(self, other_car, distance):
        self.other_car = other_car
        self.distance = float(distance)
        from srunner.osc2_stdlib.event import Event

        dis = Event.abs_distance_between_locations("ego_vehicle", "npc")
        print(other_car, dis, distance)
        if self.other_car and dis <= self.distance:
            self.flag = True
        self.__notifies(other_car, distance)

    def add_observer(self, observer):
        self.__observers.append(observer)

    def __notifies(self, *args, **kwargs):
        for o in self.__observers:
            o.update(self, *args, **kwargs)


class Observer(metaclass=ABCMeta):
    @abstractmethod
    def update(self, EventListener, *args, **kwargs):
        pass
