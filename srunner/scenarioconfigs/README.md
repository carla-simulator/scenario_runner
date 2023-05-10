本文档对config, scenario, data_bridge三个模块的功能及使用方法进行介绍

**一、config模块** 

概述：

- 对应代码：srunner/scenarioconfigs/osc2_scenario_configuration.py

- 功能：解析osc2场景描述文件，根据类型和约束的参数配置，生成标准库里相关类型对象，并设置参数。例如，描述车辆相关的类型对象ego和npc，描述路径相关的类型对象path

使用方法：

(1) 在scenario_runner.py和osc2_scenario.py文件中导入对应文件

```
from srunner.scenarioconfigs.osc2_scenario_configuration import OSC2ScenarioConfiguration
```

(2) 在scenario_runner.py文件的_run_osc2(self)函数中对OSC2ScenarioConfiguration进行初始化
```
# self._args.osc2表示输入的场景文件名称字符串
# self.client表示与carla模拟器建立连接的客户端
config = OSC2ScenarioConfiguration(self._args.osc2, self.client)
```

(3)在scenario_runner.py文件的_load_and_run_scenario(self, config)函数中，config作为输入，对OSC2Scenario进行初始化

```
scenario = OSC2Scenario(world=self.world,
                        ego_vehicles=self.ego_vehicles,
                        config=config,
                        osc2_file=self._args.osc2,
                        timeout=100000)
```
 **二、scenario模块** 

概述：

- 对应代码：/srunner/scenarios/osc2_scenario.py

- 功能：根据解析osc2场景描述文件所获得的标准库对象，抽象语法树和符号表（后两者来自语法解析阶段），建立osc2场景描述文件所对应的行为树

使用方法：

(1) 在scenario_runner.py文件中导入对应文件

```
from srunner.scenarios.osc2_scenario import OSC2Scenario
```


(2) 在scenario_runner.py文件的_load_and_run_scenario(self, config)函数中，对OSC2Scenario进行初始化

```
scenario = OSC2Scenario(world=self.world,
                        ego_vehicles=self.ego_vehicles,
                        config=config,
                        osc2_file=self._args.osc2,
                        timeout=100000)
```


(3) 以osc2_scenario.py所建立的行为树作为输入，在scenario_runner.py文件的_load_and_run_scenario(self, config)函数中，加载执行场景，并记录主车ego的行车轨迹

```
# Load scenario and run it
# self.manager是ScenarioManager类的实例化对象，对crala模拟器中场景的运行进行实时调控
self.manager.load_scenario(scenario, self.agent_instance)
self.manager.data_bridge = DataBridge(self.world)
self.manager.run_scenario()
self.manager.data_bridge.end_trace()
```
`from srunner.scenariomanager.scenario_manager import ScenarioManager`

**三、data_bridge模块**

概述：

- 功能：旨在提取场景执行时，每一帧的数据，将其写入trace.json文件中。将主车在场景中的轨迹数据文件trace.json作为交规断言的输入，进行判断。

使用方法：

(1) 在scenario_runner.py文件中导入DataBridge模块
`from data_bridge import DataBridge`

(2) 在scenario_runner.py文件中ScenarioRunner类的_load_and_run_scenario(self, config)函数中进行初始化

```
# Load scenario and run it
self.manager.load_scenario(scenario, self.agent_instance)
self.manager.data_bridge = DataBridge(self.world)
self.manager.run_scenario()
```

(3) 在srunner/scenariomanager/scenario_manager.py文件的run_scenario(self)函数中，

```
# update_ego_vehicle_start()函数根据主车ego提供的数据进行初始化
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

# self.data_bridge.update_trace()函数对carla world每个trick所提供的信息进行处理，从而获得交规断言所需要的轨迹数据。
self.data_bridge.update_trace()
```

(4) 在scenario_runner.py文件中ScenarioRunner类的_load_and_run_scenario(self, config)函数中：
```
# end_trace()函数在场景执行结束时，对轨迹数据进行更新并写入到trace.json文件中。
self.manager.data_bridge.end_trace()
```

