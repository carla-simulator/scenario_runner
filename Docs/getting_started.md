# First steps

ScenarioRunner has four modes, one per kind of scenario definition. They share the connection options and the result reporting, and differ in what they take as input and in who drives the ego vehicle.

| Mode | Option | Input | Who drives the ego |
|---|---|---|---|
| [Python scenario](#run-a-python-scenario) | `--scenario` | a configuration name from `srunner/examples/*.xml` | you, in another client |
| [OpenSCENARIO 1.x](#run-an-openscenario-file) | `--openscenario` | a `.xosc` file | the file's controller, or you |
| [OpenSCENARIO 2.0](#run-an-openscenario-20-file) | `--openscenario2` | an `.osc` file | the file's controller, or you |
| [Route](#run-a-route) | `--route` | a route file, plus `--agent` | a driving agent, inside ScenarioRunner |

This page assumes the install of [Get ScenarioRunner](getting_scenariorunner.md) and a CARLA server already running.

---
## Run a Python scenario

A Python scenario is a class in `srunner/scenarios/` plus a configuration in `srunner/examples/*.xml` that gives it a town, the ego spawn point and its parameters. You select it by the **configuration name**, such as `FollowLeadingVehicle_1`, not by the class name. `--list` prints the names this checkout can run.

```sh
python3 scenario_runner.py --scenario FollowLeadingVehicle_1 --reloadWorld --output
```

`--reloadWorld` loads the town the configuration asks for; without it the scenario refuses to start unless the server is already on that map. `--output` prints the result tables when the run ends.

ScenarioRunner does **not** drive the ego vehicle in this mode. It spawns it and waits. In a second terminal, give it a driver:

```sh
python3 manual_control.py        # drive with the keyboard
python3 manual_control.py -a     # or hand it to the autopilot
```

`manual_control.py` attaches to the vehicle whose `role_name` is `hero`. If nobody drives, the scenario's start trigger never fires and the run ends at the timeout with `GLOBAL RESULT TIMEOUT`.

To run every configuration of one class in sequence, prefix the class name with `group:`:

```sh
python3 scenario_runner.py --scenario group:FollowLeadingVehicle --reloadWorld --output
```

---
## Run an OpenSCENARIO file

An OpenSCENARIO file carries its own map, entities and ego, so there is no configuration name and no town lookup.

```sh
python3 scenario_runner.py --openscenario srunner/osc_examples/FollowLeadingVehicle.xosc --output
```

`srunner/osc_examples/` ships 16 `.xosc` files. Whether you have to drive the ego depends on the controller the file assigns to it: `external_control` waits for another client, exactly like a Python scenario, while the controllers in `srunner/scenariomanager/actorcontrols/` drive the actor from inside ScenarioRunner. `OscControllerExample.xosc` shows how a controller is assigned.

`--openscenarioparams` overrides entries of the file's global `ParameterDeclarations`:

```sh
python3 scenario_runner.py --openscenario srunner/osc_examples/FollowLeadingVehicle.xosc \
    --openscenarioparams 'leadingSpeed: 6.0' --output
```

ScenarioRunner implements a subset of the standard; [OpenSCENARIO support](openscenario_support.md) lists what is covered.

---
## Run an OpenSCENARIO 2.0 file

OpenSCENARIO 2.0 is a different language in a different file format, not a newer version of the XML. It needs `antlr4-python3-runtime` installed.

```sh
python3 scenario_runner.py --openscenario2 srunner/osc_examples/cut_in_and_slow_right.osc
```

`srunner/osc_examples/` ships 17 `.osc` files. Support is experimental; see [OpenSCENARIO 2.0](README_OpenSCENARIO_2.0.md).

---
## Run a route

Route mode drives the ego along a route given as waypoints and triggers the scenarios placed along it at their trigger points. This is the mode the [CARLA Leaderboard](https://leaderboard.carla.org/) is built on, and the only one in which ScenarioRunner drives the ego itself.

```sh
python3 scenario_runner.py --route srunner/data/routes_town10.xml --route-id 0 \
    --agent srunner/autoagents/npc_agent.py --output
```

Route mode always reloads the world, and `--agent` forces synchronous mode. Background traffic is spawned around the ego for the whole route. Route files live in `srunner/data/`:

| File | Towns |
|---|---|
| `routes_town10.xml` | `Town10HD_Opt` |
| `routes_devtest.xml`, `routes_training.xml`, `routes_validation.xml` | `Town12`, `Town13` |

Without `--route-id`, every route in the file runs in turn. See [Agent evaluation](agent_evaluation.md) for the agents and how to write one.

---
## Read the result

`--output` prints the actors, the simulation time and the criteria. `--file`, `--json` and `--junit` write the same result to disk in `--outputDir`, which a CI system can read.

```text
 > Criteria Information
╒══════════════════════╤══════════════════════╤═════════╤══════════════╤═══════════════╕
│ Actor                │ Criterion            │ Result  │ Actual Value │ Success Value │
├──────────────────────┼──────────────────────┼─────────┼──────────────┼───────────────┤
│ lincoln.mkz (id=366) │ CollisionTest (Req.) │ SUCCESS │ 0            │ 0             │
├──────────────────────┼──────────────────────┼─────────┼──────────────┼───────────────┤
│                      │ Timeout (Req.)       │ SUCCESS │ 29.38        │ 60            │
├──────────────────────┼──────────────────────┼─────────┼──────────────┼───────────────┤
│                      │ GLOBAL RESULT        │ SUCCESS │              │               │
╘══════════════════════╧══════════════════════╧═════════╧══════════════╧═══════════════╛
```

The global result is `FAILURE` if a required criterion failed, `TIMEOUT` if the timeout fired and nothing failed, `ACCEPTABLE` if a criterion ended acceptable, and `SUCCESS` otherwise.

---
## Useful options

`python3 scenario_runner.py --help` prints them all; these are the ones that come up first.

| Option | Default | Effect |
|---|---|---|
| `--host`, `--port` | `127.0.0.1`, `2000` | CARLA server to connect to. |
| `--timeout` | `120.0` | Client timeout in seconds. Raise it for large maps. |
| `--sync` | off | Run in synchronous mode, which is what reproducible runs need. |
| `--frameRate` | `20` | Tick rate in synchronous mode. |
| `--reloadWorld` | off | Reload the world, loading the scenario's town. |
| `--repetitions` | `1` | Run the scenario N times. |
| `--output`, `--file`, `--json`, `--junit` | off | Where the results go. |
| `--outputDir` | current directory | Directory for the result files. |
| `--debug` | off | Print the behaviour tree, with every node's status, on each tick. |
| `--record DIR` | | Record the run with the CARLA recorder and save the criteria, for the [metrics module](metrics_module.md). |
| `--waitForEgo` | off | Attach to an ego someone else spawned instead of spawning one. |

!!! note
    An interrupted synchronous run leaves the server in synchronous mode, which makes the world look frozen to every other client. ScenarioRunner restores asynchronous mode when it exits normally; after a crash, set `synchronous_mode` back to `False` yourself.
