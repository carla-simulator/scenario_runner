# scenario_runner
Traffic scenario definition and execution engine

## Disclaimer
The current status is work in progress and may not reflect the final API

## How To Use
The current version is designed to be used with Ubuntu 16.04, Python 2.7 and python-py-trees.
To install python-py-trees run:
```
sudo apt-get install python-py-trees
```

### Running the follow vehicle example
First of all, you need to get latest CARLA 0.9.1 release. Then you have to install the
PythonAPI:
```
easy_install ${CARLA_ROOT}/PythonAPI/carla-VERSION.egg
```

Now, you can start Carla server with Town01 from ${CARLA_ROOT}
```
./CarlaUE4.sh /Game/Carla/Maps/Town01
```

Start the example scenario (follow a leading vehicle) in an extra terminal:
```
python scenario_runner.py --scenario FollowLeadingVehicle
```

To control the ego vehicle within the scenario, open another terminal and run:
```
python manual_control.py
```
