# Getting Started Tutorial

Welcome to the ScenarioRunner for CARLA! This tutorial provides the basic steps
for getting started using the ScenarioRunner for CARLA.

Download the latest release from our [GitHub page](https://github.com/carla-simulator/scenario_runner/releases)
and extract all the contents of the package in a folder of your choice.

The release package contains the following

  * The ScenarioRunner for CARLA
  * A few example scenarios written in Python.
  * A few example scenarios using the upcoming OpenSCENARIO standard.

## Installing prerequisites
The current version is designed to be used with Ubuntu 16.04, Python 2.7 or
Python 3.5. Depending on your Python version, execute:
```bash
#Python 2.x
sudo apt remove python-networkx #if installed, remove old version of networkx
pip2 install --user py_trees==0.8.3 networkx==2.2 psutil shapely xmlschema==1.0.18
#Python 3.x
sudo apt remove python3-networkx #if installed, remove old version of networkx
pip3 install --user py_trees==0.8.3 networkx==2.2 psutil shapely xmlschema
```
Note: py-trees newer than v0.8 is *NOT* supported.

!!! important
    Please note that the Python 2 support will end in 2020.


## Running the follow vehicle example
First of all, you need to get CARLA release for the current version of ScenarioRunner (see the README for more information).
Then you have to add the CARLA Python API to the Python path:
```bash
export CARLA_ROOT=/path/to/your/carla/installation
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-<VERSION>.egg:${CARLA_ROOT}/PythonAPI/carla/agents:${CARLA_ROOT}/PythonAPI/carla
```
NOTE: ${CARLA_ROOT} needs to be replaced with your CARLA installation directory,
      and <VERSION> needs to be replaced with the correct string.
      If you build CARLA from source, the egg files maybe located in:
      ${CARLA_ROOT}/PythonAPI/dist/ instead of ${CARLA_ROOT}/PythonAPI.

Now, you can start the CARLA server from ${CARLA_ROOT}
```bash
./CarlaUE4.sh
```

Start the example scenario (follow a leading vehicle) in an extra terminal:
```bash
python scenario_runner.py --scenario FollowLeadingVehicle_1 --reloadWorld
```

If you require help or want to explore other command line parameters, start the scenario
runner as follows:
```bash
python scenario_runner.py --help
```

To control the ego vehicle within the scenario, open another terminal and run:
```bash
python manual_control.py
```

Note: If you do not wish to automatically (re-)load the CARLA world, you can
skip the command line option _--reloadWorld_

## Running all scenarios of one scenario class
Similar to the previous example, it is also possible to execute a sequence of scenarios,
that belong to the same class, e.g. the "FollowLeadingVehicle" class.

The only difference is, that you start the scenario_runner as follows:
```bash
python scenario_runner.py --scenario group:FollowLeadingVehicle
```

## Running other scenarios
A list of supported scenarios is provided in
[List of Supported Scenarios](list_of_scenarios.md). Please note that
different scenarios may take place in different CARLA towns. This has to be
respected when launching the CARLA server.

## Running scenarios using the OpenSCENARIO format
To run a scenario, which is based on the OpenSCENARIO format, please run the ScenarioRunner as follows:
```bash
python scenario_runner.py --openscenario <path/to/xosc-file>
```
Please note that the OpenSCENARIO support and the OpenSCENARIO format itself are still work in progress.
More information you can find in [OpenSCENARIO support](openscenario_support.md)

## Running route-based scenario (similar to the CARLA AD Challenge)
To run a route-based scenario, please run the ScenarioRunner as follows:
```bash
python scenario_runner.py --route <path/to/route-file> <path/to/scenario_sample_file> [route id]
```
Example:
```bash
python scenario_runner.py /scenario_runner/srunner/challenge/routes_debug.xml /scenario_runner/srunner/challenge/all_towns_traffic_scenarios1_3_4.json 0
```
If no route id is provided, all routes within the given file will be executed.

For route-based scenarios, it is also possible to use an autonomous agent to control the ego vehicle.
Several examples are provided in srunner/challenge/autoagents/. To make use of such an agent, run:
```bash
python scenario_runner.py /scenario_runner/srunner/challenge/routes_debug.xml /scenario_runner/srunner/challenge/all_towns_traffic_scenarios1_3_4.json 0 --agent /scenario_runner/srunner/challenge/autoagents/human_agent.py
```
For more information about agents, please have a look into the [agent documentation](agent_evaluation.md)

