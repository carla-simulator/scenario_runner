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
Python 3.5) and python-py-trees (v0.8.3). To install python-py-trees select one
of the following commands, depending on your Python version.
```
pip2 install --user py_trees==0.8.3 # For Python 2.x
pip3 install --user py_trees==0.8.3 # For Python 3.x
```
Note: py-trees newer than v0.8 is *NOT* supported.

Other dependencies:
In addition, you have to install Python networkx. You can install it via:
```
sudo apt-get install python-networkx
```
Please make sure that you install networkx for the Python version you want to use.


## Running the follow vehicle example
First of all, you need to get latest master branch from CARLA. Then you have to
include CARLA Python API to the Python path:
```
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla-<VERSION>.egg:${CARLA_ROOT}/PythonAPI
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


### Challenge evaluator

Define the carla root variable, where your carla instalation is located

    export CARLA_ROOT=<path_to_carla_root_folder>



Dependencies ??

    psutil , pytree 
   


TODO:
Since you already set CARLA_ROOT why do you
 need to send carla-root as a parameter ?
 
You can see the list of supported scenarios before you run the evaluation:

    python3 challenge_evaluator.py --list -a srunner/challenge/agents/DummyAgent.py
     --carla-root /home/felipe/Carla93
 
TODO: it should be just  --list ? Why do we need extra parameters on this case.


To run for instance the dummy agent on the basic scenario you should
run the following command: 


python3 challenge_evaluator.py --scenario ChallengeBasic
 -a srunner/challenge/agents/DummyAgent.py --carla-root <path_to_carla_root>




