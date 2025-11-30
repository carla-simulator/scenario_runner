#!/usr/bin/env python

# 导入所有必要的模块
import py_trees
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (LaneChange, 
                                                                       StopVehicle,
                                                                       ActorDestroy, 
                                                                       WaitForever)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
# 我们将使用更可靠的区域触发，而不是距离触发
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import InTriggerRegion
from srunner.scenarios.basic_scenario import BasicScenario
# 导入生成背景车辆的帮助函数
from srunner.tools.scenario_helper import get_waypoint_in_distance, spawn_surrounding_actors

# 1. 定义我们自己的场景类，它必须继承自 BasicScenario
class SuddenCutInScenario(BasicScenario):
    """
    一个经过整合和增强的自定义接管场景：
    1. 包含可配置数量的背景交通流，使世界更真实。
    2. 一辆危险车辆会在主车进入指定区域时，突然切入。
    3. 切入后，该危险车辆会立即紧急刹车，制造更危险的状况。
    """

    # 2. __init__: 准备阶段，从XML配置文件中读取参数
    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True, timeout=120):
        """
        初始化所有场景参数
        """
        print("----- [SCENARIO INIT] 开始初始化场景 -----")
        # 从XML读取参数，并进行类型转换
        self._target_speed = float(config.target_speed)
        self._cut_in_distance = float(config.cut_in_distance)
        # 新增参数：从XML读取触发区域和背景车辆数量
        self._trigger_region = config.trigger_region  # 格式应为 "x_min,x_max,y_min,y_max"
        self._num_background_vehicles = int(config.num_background_vehicles)
        
        print(f"[SCENARIO INIT] 参数加载成功: 目标速度={self._target_speed}, 切入距离={self._cut_in_distance}")
        print(f"[SCENARIO INIT] 触发区域='{self._trigger_region}', 背景车辆数={self._num_background_vehicles}")

        # 调用父类的构造函数，这是必须的
        super(SuddenCutInScenario, self).__init__("SuddenCutInScenario",
                                                  ego_vehicles,
                                                  config,
                                                  world,
                                                  debug_mode,
                                                  criteria_enable=criteria_enable,
                                                  timeout=timeout)

    # 3. _initialize_actors: 创建所有需要的车辆
    def _initialize_actors(self, config):
        """
        在场景开始时，创建危险车辆和背景交通流
        """
        print("----- [DEBUG] 开始执行 _initialize_actors -----")
        
        # Part 1: 创建核心的危险车辆 (用于切入和刹车)
        # 在主车前方较远处生成，给它留足准备空间
        start_wp = self.ego_vehicles[0].get_map().get_waypoint(self.ego_vehicles[0].get_location())
        waypoint_ahead, _ = get_waypoint_in_distance(start_wp, 80)
        # 优先选择右侧车道，如果没有，就在当前车道前方
        self.other_actor_wp = waypoint_ahead.get_right_lane()
        if self.other_actor_wp is None:
            print("[DEBUG] 警告: 在目标位置找不到右侧车道，将在当前车道前方生成危险车辆。")
            self.other_actor_wp = waypoint_ahead

        other_actor_transform = self.other_actor_wp.transform
        other_actor_transform.location.z += 0.1 # 稍微抬高以防生成时与地面碰撞
        
        # 请求CARLA生成车辆
        self.other_actor = CarlaDataProvider.request_new_actor(
            'vehicle.tesla.model3', other_actor_transform, rolename='other_actor')
        
        if self.other_actor is None:
            raise RuntimeError("无法生成危险车辆，请检查地图和生成点位置。")

        self.other_actor.set_simulate_physics(enabled=True)
        self.other_actors.append(self.other_actor) # 添加到列表，以便场景结束时自动销毁
        print(f"[DEBUG] 成功创建危险车辆, Actor ID: {self.other_actor.id} at {other_actor_transform.location}")

        # Part 2: 创建背景交通流
        background_actors = spawn_surrounding_actors(self.ego_vehicles[0], self._num_background_vehicles)
        for actor in background_actors:
            if actor:
                self.other_actors.append(actor)
                actor.set_autopilot(True) # 设置为自动驾驶模式

        print(f"[DEBUG] 成功创建 {len(background_actors)} 辆背景交通车辆。")
        print("----- [DEBUG] _initialize_actors 执行完毕 -----")

    # 4. _create_behavior: 定义核心行为逻辑
    def _create_behavior(self):
        """
        定义危险车辆的行为：等待触发 -> 突然切入 -> 紧急刹车
        """
        print("----- [DEBUG] 正在创建场景行为树 -----")
        # 解析从XML读取的触发区域坐标字符串
        try:
            region_coords = [float(x) for x in self._trigger_region.split(',')]
        except (ValueError, IndexError):
            raise ValueError(f"XML中的trigger_region格式错误: '{self._trigger_region}'. 应为 'x_min,x_max,y_min,y_max'")

        # 行为树是一个序列 (Sequence)，任务会按顺序执行
        sequence = py_trees.composites.Sequence("CutInAndBrakeAction")

        # 任务1: 等待主车进入指定的触发区域 (更可靠的触发方式)
        sequence.add_child(InTriggerRegion(
            self.ego_vehicles[0],
            region_coords[0], region_coords[1], # x_min, x_max
            region_coords[2], region_coords[3]  # y_min, y_max
        ))

        # 任务2: 执行向左变道切入动作
        sequence.add_child(LaneChange(
            self.other_actor, 
            speed=self._target_speed, 
            direction='left', 
            distance_same_lane=self._cut_in_distance
        ))
        
        # 任务3: 切入完成后，立即紧急刹车，并保持静止5秒
        sequence.add_child(StopVehicle(self.other_actor, 5.0))
        
        # 任务4: 保持场景继续运行，以便进行碰撞检测
        sequence.add_child(WaitForever())

        print("----- [DEBUG] 行为树创建成功 -----")
        return sequence

    # 5. _create_test_criteria: 定义成功/失败的规则
    def _create_test_criteria(self):
        """
        定义核心的测试标准：碰撞检测
        """
        return [CollisionTest(self.ego_vehicles[0])]

    # 6. __del__: 场景结束后的清理工作
    def __del__(self):
        """
        场景结束时，移除所有创建的演员
        """
        print("----- [SCENARIO END] 正在清理所有场景演员 -----")
        self.remove_all_actors()