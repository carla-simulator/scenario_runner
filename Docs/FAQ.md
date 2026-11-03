# Frequently asked questions

## `ModuleNotFoundError: No module named 'agents'`

`$CARLA_ROOT/PythonAPI/carla` is not on `PYTHONPATH`. ScenarioRunner imports the `agents` package from the CARLA source tree, and the `carla` wheel does not install it.

```sh
export PYTHONPATH=$SCENARIO_RUNNER_ROOT:$CARLA_ROOT/PythonAPI/carla:$PYTHONPATH
```

## `Scenario 'X' not supported ... Exiting`

Three causes, in order of likelihood:

*   The `type` in the configuration does not match a class name exactly.
*   The class does not derive from `BasicScenario`. Only its subclasses are discovered.
*   The file is neither in `srunner/scenarios/` nor passed with `--additionalScenario`, or `SCENARIO_RUNNER_ROOT` is unset so the lookup falls back to the working directory.

## `AttributeError` or `TypeError` inside py_trees

The installed `py-trees` does not match the one this branch pins. `ue58-dev` is written against 2.x, where a `Sequence` takes an explicit `memory` argument and a `Parallel` a policy object; `master` and the `leaderboard-*` branches pin 0.8.3. Reinstall from the checkout rather than from memory:

```sh
python3 -m pip install -r $SCENARIO_RUNNER_ROOT/requirements.txt
```

## `CARLA version 1.0 or newer required`

`scenario_runner.py` checks the server version at startup. `ue58-dev` needs a CARLA 1.0 server; a 0.10.x server is the Unreal Engine 5.5 release, for which the branch is `ue5-master`.

## Nothing happens, and the run ends with `No more scenarios .... Exiting`

In `--scenario`, `--openscenario` and `--openscenario2` modes ScenarioRunner does not drive the ego. It spawns it and waits for another client. Start one:

```sh
python3 manual_control.py        # keyboard
python3 manual_control.py -a     # autopilot
```

Without a driver the scenario's start trigger never fires and the run ends at the timeout with `GLOBAL RESULT TIMEOUT`.

## The world looks frozen to every other client

An interrupted synchronous run left the server in synchronous mode with nobody ticking it. ScenarioRunner restores asynchronous mode when it exits normally; after a crash, set it back yourself:

```py
settings = world.get_settings()
settings.synchronous_mode = False
world.apply_settings(settings)
```

## `UnicodeEncodeError` when using `--debug`

The behaviour tree is printed with non-ASCII characters. Set the output encoding:

```sh
export PYTHONIOENCODING=utf-8
```

## The town is wrong, or a blueprint is missing

The classic towns exist only as `_Opt` variants on this CARLA, so a configuration asking for `Town01` loads `Town01_Opt` and warns. Old blueprint names are translated the same way. Two-wheelers have no counterpart and the spawn fails.

## A scenario is skipped in route mode

Route mode prints `Skipping scenario 'X' due to setup error: ...` and carries on. The usual causes are a trigger point too far from the route, a spawn point already occupied, or a parameter the scenario requires missing from the route file. `--debug` prints the full traceback instead of the one-line message.
