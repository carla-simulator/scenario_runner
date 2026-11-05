# Create a new scenario

A scenario is two files: a Python class that says what happens, and an XML configuration that says where. There is no registration step — ScenarioRunner finds the class by matching the configuration's `type` attribute against the `BasicScenario` subclasses it can import.

*   __[The class](#the-class)__
*   __[The configuration](#the-configuration)__
*   __[Run it](#run-it)__
*   __[Where the files can live](#where-the-files-can-live)__
*   __[Use it inside a route](#use-it-inside-a-route)__
*   __[Rules that prevent most problems](#rules-that-prevent-most-problems)__

---
## The class

Every scenario derives from `BasicScenario` (`srunner/scenarios/basic_scenario.py`) and fills in three methods:

| Method | Returns |
|---|---|
| `_initialize_actors(config)` | Nothing. Spawns the scenario actors and appends them to `self.other_actors`. |
| `_create_behavior()` | The root of a [py_trees](https://py-trees.readthedocs.io/) behaviour tree: what the other actors do, and when the scenario ends. |
| `_create_test_criteria()` | A list of criteria, which decide pass or fail. |

In `LeadBrakes` a car drives ahead of the ego, brakes to a stop, and the scenario ends once the ego has stopped too.

```py
import py_trees

from srunner.scenariomanager.simulation_data import SimulationData
from srunner.tools.actor_factory import request_new_actor
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorDestroy,
                                                                      StopVehicle,
                                                                      WaypointFollower)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (DriveDistance,
                                                                               StandStill)
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.scenario_helper import get_waypoint_in_distance


class LeadBrakes(BasicScenario):

    """A vehicle drives ahead of the ego, brakes to a stop, and the ego must stop behind it."""

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False,
                 criteria_enable=True, timeout=60):
        self.timeout = timeout          # set before super().__init__(), which builds the tree
        self._map = SimulationData.get_map()

        # Parameters from the XML arrive as strings, in config.other_parameters
        params = config.other_parameters
        self._lead_distance = float(params.get('lead_distance', {'value': 25})['value'])
        self._lead_speed = float(params.get('lead_speed', {'value': 8})['value'])

        super().__init__("LeadBrakes", ego_vehicles, config, world, debug_mode,
                         criteria_enable=criteria_enable)

    def _initialize_actors(self, config):
        ego_wp = self._map.get_waypoint(config.trigger_points[0].location)
        lead_wp, _ = get_waypoint_in_distance(ego_wp, self._lead_distance)
        transform = lead_wp.transform
        transform.location.z += 0.5
        lead = request_new_actor('vehicle.tesla.model3', transform)
        self.other_actors.append(lead)

    def _create_behavior(self):
        lead, ego = self.other_actors[0], self.ego_vehicles[0]

        # The lead drives along its lane until it has covered 40 m
        drive = py_trees.composites.Parallel(
            "LeadDrives", policy=py_trees.common.ParallelPolicy.SuccessOnOne())
        drive.add_child(WaypointFollower(lead, self._lead_speed))
        drive.add_child(DriveDistance(lead, 40))

        root = py_trees.composites.Sequence("LeadBrakes", memory=True)
        root.add_child(DriveDistance(ego, 2, name="EgoStarts"))
        root.add_child(drive)
        root.add_child(StopVehicle(lead, 1.0))                   # full brake
        root.add_child(StandStill(ego, name="EgoStopped", duration=1))
        root.add_child(ActorDestroy(lead))
        return root

    def _create_test_criteria(self):
        return [CollisionTest(self.ego_vehicles[0])]

    def __del__(self):
        self.remove_all_actors()
```

A behaviour tree is built from **composites** and **leaves**. A `Sequence` runs its children one after the other; a `Parallel` runs them together and succeeds when one child does (`ParallelPolicy.SuccessOnOne()`) or when all do (`ParallelPolicy.SuccessOnAll()`). The leaves are the atomics:

| Module | Contains | Examples |
|---|---|---|
| `atomic_behaviors.py` | actions on actors | `WaypointFollower`, `KeepVelocity`, `StopVehicle`, `LaneChange`, `ActorTransformSetter`, `ActorDestroy` |
| `atomic_trigger_conditions.py` | conditions that succeed when something happens | `InTriggerDistanceToVehicle`, `InTriggerDistanceToNextIntersection`, `DriveDistance`, `StandStill` |
| `atomic_criteria.py` | criteria, evaluated for the whole run | `CollisionTest`, `KeepLaneTest`, `OnSidewalkTest`, `RunningRedLightTest`, `RunningStopTest`, `RouteCompletionTest` |

!!! important
    This branch pins `py-trees==2.6.0`. A `Sequence` takes an explicit `memory` argument and a `Parallel` a policy **object**. The `master` and `leaderboard-*` branches pin `0.8.3`, where the same code is `Sequence("name")` and `policy=ParallelPolicy.SUCCESS_ON_ONE`. Trees written for one version do not load on the other.

`BasicScenario` wraps your tree in a root `Parallel` that also holds the criteria, a timeout and a node that applies the actors' controls, and puts a start trigger in front of it. The scenario therefore ends as soon as your tree succeeds or the timeout fires.

---
## The configuration

The configuration gives the scenario a town and an ego spawn point, which is also the scenario's trigger point. Any element other than `ego_vehicle` and the recognised ones becomes an entry of `config.other_parameters`.

```xml
<?xml version="1.0"?>
<scenarios>
    <scenario name="LeadBrakes_1" type="LeadBrakes" town="Town01_Opt">
        <ego_vehicle x="107" y="133" z="0.5" yaw="0" model="vehicle.lincoln.mkz" />
        <lead_distance value="25" />
        <lead_speed value="8" />
    </scenario>
</scenarios>
```

`name` is what you pass to `--scenario`; `type` has to match the class name exactly, or the run ends with `Scenario 'X' not supported`.

A `<weather>` element sets the weather of the run, and `<other_actor>` declares an actor the scenario can read from `config.other_actors`.

---
## Run it

```sh
python3 scenario_runner.py --scenario LeadBrakes_1 \
    --additionalScenario lead_brakes.py --configFile LeadBrakes.xml \
    --reloadWorld --output
```

Then drive the ego from another terminal with `manual_control.py`, as in [First steps](getting_started.md).

---
## Where the files can live

Dropping the two files into `srunner/scenarios/` and `srunner/examples/` makes them available with no extra flags, but it means editing the checkout. To keep them outside it, pass them instead:

| Flag | Takes |
|---|---|
| `--additionalScenario` | a `.py` file, a folder of them, or several separated by commas |
| `--configFile` | an `.xml` file, a folder of them, or several separated by commas |

Both flags can be repeated. A whole directory of scenarios is therefore one flag:

```sh
python3 scenario_runner.py --scenario LeadBrakes_1 \
    --additionalScenario ~/my_scenarios --configFile ~/my_configs \
    --reloadWorld --output
```

The classes found this way are available to `--route` as well as to `--scenario`.

---
## Use it inside a route

Reference the class by its `type` in the route file's `scenarios` block, with a `trigger_point`:

```xml
<scenario name="LeadBrakes_1" type="LeadBrakes">
    <trigger_point x="107" y="133" z="0.5" yaw="0"/>
</scenario>
```

For this to work the scenario has to place its actors relative to `config.trigger_points[0]`, as `_initialize_actors()` above does, rather than at fixed coordinates. A scenario written that way runs anywhere on any route.

---
## Rules that prevent most problems

*   **Derive from `BasicScenario`.** Class discovery keeps only its subclasses, so a class that does not derive from it is invisible, with the same "not supported" message as a typo in `type`.
*   **Spawn with `request_new_actor()`** from `srunner.tools.actor_factory`, not `world.spawn_actor()`, and append what it returns to `self.other_actors` so `remove_all_actors()` destroys it.
*   **Append every actor to `self.other_actors`**, or it outlives the scenario.
*   **Make sure the tree can end.** Every child of your root `Sequence` has to succeed at some point, or every run ends at the timeout. `--debug` prints the tree each tick and shows which node is still `RUNNING`.
*   **Set `self.timeout` before calling `super().__init__()`**, which is what builds the tree.
