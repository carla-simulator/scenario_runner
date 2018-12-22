# Getting Started Tutorial

!!! important
    This tutorial refers to the latest versions of CARLA (at least 0.9.2)

Welcome to the ScenarioRunner for CARLA! This tutorial provides the basic steps
for getting started using the ScenarioRunner for CARLA.

Download the latest release from our GitHub page and extract all the contents of
the package in a folder of your choice.

The release package contains the following

  * The ScenarioRunner for CARLA
  * A few example scenarios written in Python.

## Installing prerequisites
The current version is designed to be used with Ubuntu 16.04, Python 2.7 (or
Python 3.5) and python-py-trees. To install python-py-trees select one of the
following commands, depending on your Python version.
```
pip2 install --user py_trees # For Python 2.x
pip3 install --user py_trees # For Python 3.x
```

## Running the follow vehicle example
First of all, you need to get latest master branch from CARLA. Then you have to
include CARLA Python API to the Python path:
```
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla-<VERSION>.egg
```

Now, you can start the CARLA server with Town01 from ${CARLA_ROOT}
```
./CarlaUE4.sh /Game/Carla/Maps/Town01 -benchmark -fps=20 -windowed
```
Please note that using the benchmark mode with a defined frame rate is very
important to achieve a deterministic behavior.

Start the example scenario (follow a leading vehicle) in an extra terminal:
```
python scenario_runner.py --scenario FollowLeadingVehicle
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

## Running other scenarios
A list of supported scenarios is provided in
[List of Supported Scenarios](list_of_scenarios.md). Please note that
different scenarios may take place in different CARLA towns. This has to be
respected when launching the CARLA server.
