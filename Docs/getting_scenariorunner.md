# Get ScenarioRunner

ScenarioRunner is a pure Python client of the CARLA server. Nothing is added to the server and nothing has to be compiled: clone the branch that matches your CARLA, install the requirements, and point `PYTHONPATH` at the two trees it imports.

*   __[Pick the branch](#pick-the-branch)__
*   __[Install](#install)__
*   __[Set the environment](#set-the-environment)__
*   __[Check the installation](#check-the-installation)__
*   __[Update](#update)__

---
## Pick the branch

Each branch targets one CARLA line. The scenarios, town names and blueprint names differ between them, so a mismatched pair fails at the first spawn even when it starts.

| Branch | CARLA | Notes |
|---|---|---|
| `ue58-dev` | __1.0__ | The current branch, on Unreal Engine 5.8. 96 configurations across `Town01_Opt`–`Town05_Opt` and `Town10HD_Opt`. Pins `py-trees==2.6.0`. |
| `ue5-master` | 0.10.0 | The Unreal Engine 5.5 release. `Town10HD_Opt` only. |
| `master` | 0.9.14 – 0.9.16 | The Unreal Engine 4.26 line. |
| `leaderboard-1.0`, `leaderboard-2.0`, `leaderboard-2.1` | per the Leaderboard guide | Pinned to a Leaderboard version. |

`scenario_runner.py` refuses to start against a server older than the `MIN_CARLA_VERSION` of its branch, which is `1.0` on `ue58-dev`.

!!! important
    The `py-trees` pin is not interchangeable. `ue58-dev` is written against the 2.x API, where a `Sequence` takes an explicit `memory` argument and a `Parallel` takes a policy object; the other branches pin `0.8.3`. Installing the wrong one gives `AttributeError` or `TypeError` inside py_trees as soon as a tree is built.

---
## Install

__1.__ Install the CARLA Python client into the interpreter you will use. Use the wheel that came with your CARLA, so that the client and server versions match.

__2.__ Clone the branch and install its requirements into that same interpreter.

```sh
git clone -b ue58-dev --single-branch https://github.com/carla-simulator/scenario_runner.git
cd scenario_runner
python3 -m pip install -r requirements.txt
```

The requirements cover every mode. `xmlschema` validates OpenSCENARIO 1.x files, `antlr4-python3-runtime` is needed only for OpenSCENARIO 2.0 (`--openscenario2`), and `matplotlib` and `graphviz` only for the [metrics module](metrics_module.md).

---
## Set the environment

```sh
export CARLA_ROOT=/path/to/carla
export SCENARIO_RUNNER_ROOT=/path/to/scenario_runner
export PYTHONPATH=$SCENARIO_RUNNER_ROOT:$CARLA_ROOT/PythonAPI/carla:$PYTHONPATH
```

Both paths are needed, for different reasons:

*   `SCENARIO_RUNNER_ROOT` is read at run time to find `srunner/scenarios/*.py` and `srunner/examples/*.xml`, and to resolve the `--record` directory. Without it those lookups fall back to the working directory and no scenario is found.
*   `$CARLA_ROOT/PythonAPI/carla` has to be on `PYTHONPATH` because ScenarioRunner imports the `agents` package (`agents.navigation.*`), which lives in the CARLA source tree and is **not** installed by the `carla` wheel.

If `scenario_runner.py` fails with `ModuleNotFoundError: No module named 'agents'`, the second path is missing.

---
## Check the installation

Listing the configurations needs no server:

```sh
python3 $SCENARIO_RUNNER_ROOT/scenario_runner.py --list
```

```text
Currently the following scenarios are supported:
FollowLeadingVehicle_1
FollowLeadingVehicle_2
...
VehicleTurningRight_1
...
```

`ue58-dev` lists 96 configurations. Then start a CARLA server and run one of them, as described in [First steps](getting_started.md).

---
## Update

```sh
cd $SCENARIO_RUNNER_ROOT
git pull
python3 -m pip install -r requirements.txt
```

Re-run the requirements install after a pull. A branch that moves its `py-trees` pin leaves the old version installed, and the failure then appears later, inside a behaviour tree, rather than at startup.
