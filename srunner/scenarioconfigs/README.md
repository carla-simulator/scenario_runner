This document introduces the functions and usage methods of the three modules config, scenario and data bridge

**1. config**

- Code：srunner/scenarioconfigs/osc2_scenario_configuration.py

- Function: Parses the osc2 scenario description file, generates type objects in the standard library based on the type and constraint parameters, and sets parameters, for example, ego and npc related to vehicles and path related to paths

Usage：

(1) Import the corresponding files in the scenario runner.py and osc2 Scenario.py files

```
from srunner.scenarioconfigs.osc2_scenario_configuration import OSC2ScenarioConfiguration
```

(2) In the scenario runner. Py files run osc2 (self) function to initialize the OSC2ScenarioConfiguration
```
# self._args.osc2: The input scene file name string
# self.client: The client that connects to the carla simulator
config = OSC2ScenarioConfiguration(self._args.osc2, self.client)
```

(3)OSC2Scenario is initialized in the load and run scenario(self, config) function of the scenario runner.py file with config as input

```
scenario = OSC2Scenario(world=self.world,
                        ego_vehicles=self.ego_vehicles,
                        config=config,
                        osc2_file=self._args.osc2,
                        timeout=100000)
```
 **2. scenario**

- Code：/srunner/scenarios/osc2_scenario.py

- Function：The behavior tree corresponding to the osc2 scene description file is created based on the standard library objects obtained by parsing the osc2 scene description file, the abstract syntax tree, and the symbol table (the latter two are from the syntax parsing phase)

Usage:

(1) Import the corresponding file in the scenario runner.py file

```
from srunner.scenarios.osc2_scenario import OSC2Scenario
```


(2) OSC2Scenario is initialized in the load and run scenario(self, config) functions of scenario runner.py file

```
scenario = OSC2Scenario(world=self.world,
                        ego_vehicles=self.ego_vehicles,
                        config=config,
                        osc2_file=self._args.osc2,
                        timeout=100000)
```


(3) The behavior tree established by osc2_scenario.py is used as input, and in the _Load_AND_RUN_SCENARIO (interfig) function of the SCENARIO_Runner.py file, loads the execution scene, and records the driving trajectory of the main car EGO

```
# Load scenario and run it
# self.manager is an instantiated object of the SCENARIOMANAGER class. In real -time regulation of the operation of the scene in the Crala simulator
self.manager.load_scenario(scenario, self.agent_instance)
self.manager.data_bridge = DataBridge(self.world)
self.manager.run_scenario()
self.manager.data_bridge.end_trace()
```
`from srunner.scenariomanager.scenario_manager import ScenarioManager`

**3. data_bridge**

- Function：The purpose is to extract the data of each frame when the scene is executed, write it into the trace.json file, and use the trace data file of the main vehicle in the scene as the input of the traffic regulation assertion for judgment

Usage:

(1) Import the DataBridge module in the scenario runner.py file
`from data_bridge import DataBridge`

(2) Initialization is done in the load and run scenario(self, config) functions of the ScenarioRunner class in the scenario runner.py file

```
# Load scenario and run it
self.manager.load_scenario(scenario, self.agent_instance)
self.manager.data_bridge = DataBridge(self.world)
self.manager.run_scenario()
```

(3) srunner/scenariomanager/scenario_manager.py -> run_scenario(self)

```
self.data_bridge.update_ego_vehicle_start(self.ego_vehicles[0])

while self._running:
timestamp = None
world = CarlaDataProvider.get_world()
if world:
snapshot = world.get_snapshot()
if snapshot:
timestamp = snapshot.timestamp
if timestamp:
self._tick_scenario(timestamp)
self.data_bridge.update_trace()
```

(4) scenario_runner.py -> ScenarioRunner -> _load_and_run_scenario(self, config)
```
self.manager.data_bridge.end_trace()
```

