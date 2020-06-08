import carla
import os

client = carla.Client('127.0.0.1',2000)

world = client.get_world()
actors = world.get_actors()

for actor in actors:
    if 'traffic_light' in actor.type_id:
        actor.freeze(True)
        actor.set_state(carla.TrafficLightState.Unknown)

client.start_recorder("{}/{}.log".format(os.getenv('SCENARIO_RUNNER_ROOT', "./"), 'Frozen'))
world.wait_for_tick()
world.wait_for_tick()
world.wait_for_tick()
world.wait_for_tick()
client.stop_recorder()
info = client.show_recorder_file_info("{}/{}.log".format(os.getenv('SCENARIO_RUNNER_ROOT', "./"), 'Frozen'), True)
print(info)