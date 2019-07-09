# Getting Started Tutorial

!!! important
    This tutorial refers to the latest versions of CARLA (at least 0.9.5)

Welcome to the ScenarioRunner for CARLA! This tutorial provides the basic steps
for getting started using the ScenarioRunner for CARLA.

Download the latest release from our GitHub page and extract all the contents of
the package in a folder of your choice.

The release package contains the following

  * The ScenarioRunner for CARLA
  * A few example scenarios written in Python.

## Installing prerequisites
The current version is designed to be used with Ubuntu 16.04, Python 2.7 or
Python 3.5. Depending on your Python version, execute:
```


#Python 2.x
sudo apt remove python-networkx #if installed, remove old version of networkx
pip2 install --user py_trees==0.8.3 networkx==2.2 psutil shapely xmlschema
#Python 3.x
sudo apt remove python3-networkx #if installed, remove old version of networkx
pip3 install --user py_trees==0.8.3 networkx==2.2 psutil shapely xmlschema
```
Note: py-trees newer than v0.8 is *NOT* supported.



## Running the follow vehicle example
First of all, you need to get latest master branch from CARLA. Then you have to
include CARLA Python API to the Python path:
```
export CARLA_ROOT=/path/to/your/carla/installation
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-<VERSION>.egg:${CARLA_ROOT}/PythonAPI/carla/agents:${CARLA_ROOT}/PythonAPI/carla
```
NOTE: ${CARLA_ROOT} needs to be replaced with your CARLA installation directory,
      and <VERSION> needs to be replaced with the correct string.
      If you build CARLA from source, the egg files maybe located in:
      ${CARLA_ROOT}/PythonAPI/dist/ instead of ${CARLA_ROOT}/PythonAPI.

Now, you can start the CARLA server with Town01 from ${CARLA_ROOT}
```
./CarlaUE4.sh /Game/Carla/Maps/Town01 -benchmark -fps=20 -windowed
```
Please note that using the benchmark mode with a defined frame rate is very
important to achieve a deterministic behavior.

Start the example scenario (follow a leading vehicle) in an extra terminal:
```
python scenario_runner.py --scenario FollowLeadingVehicle_1
```

If you require help or want to explore other command line parameters, start the scenario
runner as follows:
```
python scenario_runner.py --help
```

To control the ego vehicle within the scenario, open another terminal and run:
```
python manual_control.py
```

## Running all scenarios of one scenario class
Similar to the previous example, it is also possible to execute a sequence of scenarios,
that belong to the same class, e.g. the "FollowLeadingVehicle" class.

The only difference is, that you start the scenario_runner as follows:
```
python scenario_runner.py --scenario group:FollowLeadingVehicle
```

## Running other scenarios
A list of supported scenarios is provided in
[List of Supported Scenarios](list_of_scenarios.md). Please note that
different scenarios may take place in different CARLA towns. This has to be
respected when launching the CARLA server.

## Running scenarios using the OpenScenario format
To run a scenario, which is based on the OpenScenario format, please run the scenario runner as follows:
```
python scenario_runner.py --openscenario <path/to/xosc-file>
```
Please note that the OpenScenario support and the OpenScenario format itself are still work in progress.
More information you can find in [OpenScenario support](openscenario_support.md)
