# Agent evaluation

In route mode ScenarioRunner drives the ego itself, through a **driving agent**: a Python class that receives sensor data every tick and returns a `carla.VehicleControl`. This is how a complete driving stack is evaluated against a route and the scenarios placed along it, and it is the mechanism the [CARLA Leaderboard](https://leaderboard.carla.org/) is built on.

In every other mode ScenarioRunner only spawns the ego and waits for another client to drive it.

*   __[Run a route with an agent](#run-a-route-with-an-agent)__
*   __[The agents that ship](#the-agents-that-ship)__
*   __[Write your own agent](#write-your-own-agent)__
*   __[Route criteria](#route-criteria)__

---
## Run a route with an agent

```sh
python3 scenario_runner.py --route srunner/data/routes_town10.xml --route-id 0 \
    --agent srunner/autoagents/npc_agent.py --output
```

`--agent` is accepted only with `--route`. It forces synchronous mode, and route mode always reloads the world. Background traffic is spawned around the ego for the whole route.

`--agentConfig` passes a file path through to the agent's `setup()`; what it contains is up to the agent.

!!! important
    ScenarioRunner derives the class name from the file name: `npc_agent.py` must define `NpcAgent`, `my_agent.py` must define `MyAgent`. A mismatch gives an import error rather than a useful message.

---
## The agents that ship

They live in `srunner/autoagents/`.

| File | Class | Behaviour |
|---|---|---|
| `npc_agent.py` | `NpcAgent` | Follows the route with a `BasicAgent` from the CARLA `agents` package. The reference check that a route works. |
| `human_agent.py` | `HumanAgent` | Keyboard control in a pygame window. Its configuration file can record the controls to JSON and play them back. |
| `dummy_agent.py` | `DummyAgent` | Returns an empty control. The template to copy. |
| `ros_agent.py` | `RosAgent` | Publishes the sensor data to ROS and reads back the control. See [ROS agent](ros_agent.md). |

`agent_wrapper.py` and `sensor_interface.py` are the machinery around them: the wrapper spawns and attaches the sensors the agent asks for and validates them, and the sensor interface collects their data into the dictionary the agent is given.

---
## Write your own agent

An agent derives from `AutonomousAgent` (`srunner/autoagents/autonomous_agent.py`) and implements four methods.

| Method | Called | Does |
|---|---|---|
| `setup(path_to_conf_file)` | once, from `__init__` | Initialises the agent. Receives whatever `--agentConfig` points at. |
| `sensors()` | once, before the route starts | Returns the sensor suite, as a list of dictionaries. |
| `run_step(input_data, timestamp)` | every tick | Returns a `carla.VehicleControl`. |
| `destroy()` | at the end of the route | Releases anything the agent holds. |

`sensors()` returns one dictionary per sensor, with its type, its position relative to the ego, its attributes and an `id` you choose. That `id` is the key under which its data arrives in `run_step`:

```py
def sensors(self):
    return [
        {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.0, 'z': 1.60,
         'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
         'width': 300, 'height': 200, 'fov': 100, 'id': 'Center'},
        {'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': 0.0, 'z': 1.60,
         'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'id': 'LIDAR'},
        {'type': 'sensor.other.gnss', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'id': 'GPS'},
    ]
```

A minimal agent that drives slowly forward:

```py
import carla
from srunner.autoagents.autonomous_agent import AutonomousAgent


class MyAgent(AutonomousAgent):

    def setup(self, path_to_conf_file):
        self._target_speed = 20

    def sensors(self):
        return [{'type': 'sensor.other.gnss', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'id': 'GPS'}]

    def run_step(self, input_data, timestamp):
        control = carla.VehicleControl()
        control.throttle = 0.3
        control.steer = 0.0
        control.brake = 0.0
        return control

    def destroy(self):
        pass
```

The route itself is given to the agent before the run starts, as `self._global_plan` (GPS coordinates with a high-level command per point) and `self._global_plan_world_coord` (the same in world coordinates). Both are downsampled from the dense interpolated route, so they are waypoints to aim at rather than a path to track exactly.

The same four methods are the Leaderboard's agent interface, so an agent written here runs there without changes.

---
## Route criteria

A route run evaluates the route criteria for the whole route, on top of whatever criteria each triggered scenario adds:

| Criterion | Fails when |
|---|---|
| `RouteCompletionTest` | The route is not completed. Reported as a percentage. |
| `OutsideRouteLanesTest` | The ego drives outside the lanes of its route. Reported as a percentage of the distance. |
| `CollisionTest` | The ego collides with anything. |
| `RunningRedLightTest` | The ego crosses a red light. |
| `RunningStopTest` | The ego fails to stop at a stop sign. |
| `MinimumSpeedRouteTest` | The ego is much slower than the surrounding traffic. |
| `InRouteTest` | The ego leaves the route by more than the allowed distance. |
| `ActorBlockedTest` | The ego stays still too long. |
| `ScenarioTimeoutTest` | A triggered scenario does not finish in time. |

The run ends early when `InRouteTest` or `ActorBlockedTest` fails, since the route can no longer be completed. As in any other mode, the result is printed with `--output` and written with `--file`, `--json` or `--junit`.
