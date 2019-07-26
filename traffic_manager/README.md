# Traffic Manager


##Getting Traffic Manager

Use git clone or download the project from [Carla-simulator/scenario_runner](https://github.com/carla-simulator/scenario_runner) and switch to branch traffic_manager.


## Using Traffic Manager

Go to ${scenario_runner_ROOT}/traffic_manager/source

Set an environment variable, LIBCARLA_LOCATION to where your build of libcarla resides.

Now you can start the CARLA server from ${CARLA_ROOT} in a new terminal and then spwan some vehicles (You can use the spawn_npc.py file in PythonAPI/examples but you will need to edit the code and set autopilot to false).

Then create a new directory "build" in ${scenario_runner_ROOT}/traffic_manager/source

```

mkdir build
cd build
cmake..
make
./traffic_manager

```
