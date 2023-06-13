from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
import json


class ActorEncoder(json.JSONEncoder):
    def default(self, obj):
        return obj.__dict__


class ActorInfo:
    def __init__(self, transform, speed):
        self.world_position = {
            'x': transform.location.x,
            'y': transform.location.y,
            'z': transform.location.z,
            'pitch': transform.rotation.pitch,
            'yaw': transform.rotation.yaw,
            'roll': transform.rotation.roll,
        }
        self.speed = speed

    def __str__(self):
        return json.dumps(self.__dict__)


class ActorsInfo:

    @staticmethod
    def getstate():

        state_json = dict()
        actors = dict(CarlaDataProvider.get_actors())
        for _, actor in actors.items():
            state_json[actor.attributes['role_name']] = ActorInfo(
                CarlaDataProvider.get_transform(actor),
                CarlaDataProvider.get_velocity(actor)
            )
        return json.dumps(state_json, cls=ActorEncoder)
