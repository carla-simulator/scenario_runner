# Traffic Manager

## Using Traffic Manager

Start the CARLA server from ${CARLA_ROOT} in a new terminal and then spwan some vehicles (Note: You can use the spawn_npc.py file in ${CARLA_ROOT}/PythonAPI/examples but you will need to edit the code and set autopilot to false).

Set an environment variable, LIBCARLA_LOCATION to where your build of libcarla resides.

Go to ${TRAFFIC_MANAGER_ROOT}/source.

Create a new directory "build" in ${scenario_runner_ROOT}/traffic_manager/source, build and run.

```

mkdir build
cd build
cmake..
make
./traffic_manager

```
