from __future__ import print_function

import copy
import math
import operator
import random
import re
import sys
from typing import List, Tuple

import py_trees
from agents.navigation.global_route_planner import GlobalRoutePlanner

from srunner.osc2.ast_manager import ast_node
from srunner.osc2.ast_manager.ast_vistor import ASTVisitor

# OSC2
from srunner.osc2.symbol_manager.method_symbol import MethodSymbol
from srunner.osc2.symbol_manager.parameter_symbol import ParameterSymbol
from srunner.osc2.utils.log_manager import (LOG_INFO, LOG_ERROR, LOG_WARNING)
from srunner.osc2.utils.relational_operator import RelationalOperator
from srunner.osc2_dm.physical_types import Physical, Range

# from sqlalchemy import true
# from srunner.osc2_stdlib import event, variables
from srunner.osc2_stdlib.modifier import (
    AccelerationModifier,
    ChangeLaneModifier,
    ChangeSpeedModifier,
    LaneModifier,
    PositionModifier,
    SpeedModifier,
)

# OSC2
from srunner.scenarioconfigs.osc2_scenario_configuration import (
    OSC2ScenarioConfiguration,
)
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (
    ActorTransformSetter,
    ChangeTargetSpeed,
    LaneChange,
    UniformAcceleration,
    WaypointFollower,
    calculate_distance,
)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (
    IfTriggerer,
    TimeOfWaitComparison,
)
from srunner.scenariomanager.timer import TimeOut
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.openscenario_parser import oneshot_with_check
from srunner.tools.osc2_helper import OSC2Helper


def para_type_str_sequence(config, arguments, line, column, node):
    retrieval_name = ""
    if isinstance(arguments, List):
        for arg in arguments:
            if isinstance(arg, Tuple):
                if isinstance(arg[1], int):
                    retrieval_name = retrieval_name + "#int"
                elif isinstance(arg[1], float):
                    retrieval_name = retrieval_name + "#float"
                elif isinstance(arg[1], str):
                    retrieval_name = retrieval_name + "#str"
                elif isinstance(arg[1], bool):
                    retrieval_name = retrieval_name + "#bool"
                elif isinstance(arg[1], Physical):
                    physical_type_name = OSC2Helper.find_physical_type(
                        config.physical_dict, arg[1].unit.physical.si_base_exponent
                    )

                    if physical_type_name is None:
                        pass
                    else:
                        physical_type = (
                            node.get_scope().resolve(physical_type_name).name
                        )
                        retrieval_name += "#" + physical_type
                else:
                    pass
            elif isinstance(arg, str):
                retrieval_name = retrieval_name + arg.split(".", 1)[-1]
            else:
                pass
    elif isinstance(arguments, Tuple):
        if isinstance(arguments[1], int):
            retrieval_name = retrieval_name + "#int"
        elif isinstance(arguments[1], float):
            retrieval_name = retrieval_name + "#float"
        elif isinstance(arguments[1], str):
            retrieval_name = retrieval_name + "#str"
        elif isinstance(arguments[1], bool):
            retrieval_name = retrieval_name + "#bool"
        elif isinstance(arguments[1], Physical):
            physical_type_name = OSC2Helper.find_physical_type(
                config.physical_dict, arguments[1].unit.physical.si_base_exponent
            )

            if physical_type_name is None:
                pass
            else:
                physical_type = node.get_scope().resolve(physical_type_name).name
                retrieval_name += "#" + physical_type
        else:
            pass
    elif isinstance(arguments, int):
        retrieval_name = retrieval_name + "#int"
    elif isinstance(arguments, float):
        retrieval_name = retrieval_name + "#float"
    elif isinstance(arguments, str):
        retrieval_name = retrieval_name + "#str"
    elif isinstance(arguments, bool):
        retrieval_name = retrieval_name + "#bool"
    elif isinstance(arguments, Physical):
        physical_type_name = OSC2Helper.find_physical_type(
            config.physical_dict, arguments.unit.physical.si_base_exponent
        )

        if physical_type_name is None:
            pass
        else:
            physical_type = node.get_scope().resolve(physical_type_name).name
            retrieval_name += "#" + physical_type
    else:
        pass
    return retrieval_name


def process_speed_modifier(
    config, modifiers, duration: float, all_duration: float, father_tree
):
    if not modifiers:
        return

    for modifier in modifiers:
        actor_name = modifier.get_actor_name()

        if isinstance(modifier, SpeedModifier):
            # en_value_mps() The speed unit in Carla is m/s, so the default conversion unit is m/s
            target_speed = modifier.get_speed().gen_physical_value()
            # target_speed = float(modifier.get_speed())*0.27777778
            actor = CarlaDataProvider.get_actor_by_name(actor_name)
            car_driving = WaypointFollower(actor, target_speed)
            # car_driving.set_duration(duration)

            father_tree.add_child(car_driving)

            car_config = config.get_car_config(actor_name)
            car_config.set_arg({"target_speed": target_speed})
            LOG_WARNING(
                f"{actor_name} car speed will be set to {target_speed * 3.6} km/h"
            )

            # # _velocity speed, go straight down the driveway, and will hit the wall
            # keep_speed = KeepVelocity(actor, target_speed, duration=father_duration.num)
        elif isinstance(modifier, ChangeSpeedModifier):
            # speed_delta indicates the increment of velocity
            speed_delta = modifier.get_speed().gen_physical_value()
            speed_delta = speed_delta * 3.6
            current_car_conf = config.get_car_config(actor_name)
            current_car_speed = current_car_conf.get_arg("target_speed")
            current_car_speed = current_car_speed * 3.6
            target_speed = current_car_speed + speed_delta
            LOG_WARNING(
                f"{actor_name} car speed will be changed to {target_speed} km/h"
            )

            actor = CarlaDataProvider.get_actor_by_name(actor_name)
            change_speed = ChangeTargetSpeed(actor, target_speed)

            car_driving = WaypointFollower(actor)
            # car_driving.set_duration(duration)

            father_tree.add_child(change_speed)
            father_tree.add_child(car_driving)
        elif isinstance(modifier, AccelerationModifier):
            current_car_conf = config.get_car_config(actor_name)
            current_car_speed = current_car_conf.get_arg("target_speed")
            accelerate_speed = modifier.get_accelerate().gen_physical_value()
            target_velocity = current_car_speed + accelerate_speed * duration
            actor = CarlaDataProvider.get_actor_by_name(actor_name)
            start_time = all_duration - duration
            uniform_accelerate_speed = UniformAcceleration(
                actor, current_car_speed, target_velocity, accelerate_speed, start_time
            )
            print("END ACCELERATION")
            car_driving = WaypointFollower(actor)

            father_tree.add_child(uniform_accelerate_speed)
            father_tree.add_child(car_driving)
        else:
            LOG_WARNING("not implement modifier")


def process_location_modifier(config, modifiers, duration: float, father_tree):
    # position([distance: ]<distance> | time: <time>, [ahead_of: <car> | behind: <car>], [at: <event>])
    # lane([[lane: ]<lane>][right_of | left_of | same_as: <car>] | [side_of: <car>, side: <av-side>][at: <event>])
    """
    Implementation idea: First determine the lane through lane,
    and then determine the position in the lane through position.
    If the trigger event is start, then directly set the strasform.
    If it is end, calculate the distance between the current position and the end position,
    and then divide by duration to get the speed of the car.
    @TODO supplement absolute positioning.
    """
    if not modifiers:
        return

    for modifier in modifiers:
        if isinstance(modifier, ChangeLaneModifier):
            lane_changes = modifier.get_lane_changes()
            av_side = modifier.get_side()
            print(f"The car changes lanes to the {av_side} for {lane_changes} lanes.")
            npc_name = modifier.get_actor_name()
            actor = CarlaDataProvider.get_actor_by_name(npc_name)
            lane_change = LaneChange(
                actor, speed=None, direction=av_side, lane_changes=lane_changes
            )
            continue_drive = WaypointFollower(actor)
            father_tree.add_child(lane_change)
            father_tree.add_child(continue_drive)
            print("END of change lane--")
            return
    # start
    # Deal with absolute positioning vehicles first，such as lane(1, at: start)
    event_start = [
        m
        for m in modifiers
        if m.get_trigger_point() == "start" and m.get_refer_car() is None
    ]

    for m in event_start:
        car_name = m.get_actor_name()
        wp = CarlaDataProvider.get_waypoint_by_laneid(m.get_lane_id())
        if wp:
            actor = CarlaDataProvider.get_actor_by_name(car_name)
            actor_visible = ActorTransformSetter(actor, wp.transform)
            father_tree.add_child(actor_visible)

            car_config = config.get_car_config(car_name)
            car_config.set_arg({"init_transform": wp.transform})
            LOG_INFO(
                f"{car_name} car init position will be set to {wp.transform.location}, roadid = {wp.road_id}, laneid={wp.lane_id}, s = {wp.s}"
            )
        else:
            raise RuntimeError(f"no valid position to spawn {car_name} car")

    # Handle relative positioning vehicles
    start_group = [
        m
        for m in modifiers
        if m.get_trigger_point() == "start" and m.get_refer_car() is not None
    ]

    init_wp = None
    npc_name = None

    for modifier in start_group:
        npc_name = modifier.get_actor_name()
        # location reprents npc at ego_vehicle left, right, same, ahead
        relative_car_name, location = modifier.get_refer_car()
        relative_car_conf = config.get_car_config(relative_car_name)
        relative_car_location = relative_car_conf.get_transform().location

        relative_wp = CarlaDataProvider.get_map().get_waypoint(relative_car_location)

        if init_wp is None:
            init_wp = relative_wp

        if location == "left_of":
            temp_lane = init_wp.get_left_lane()
            if temp_lane:
                init_wp = temp_lane

        elif location == "right_of":
            temp_lane = init_wp.get_right_lane()
            if temp_lane:
                init_wp = temp_lane
        elif location == "same_as":
            # Same lane
            pass
        elif location in ('ahead_of', 'behind'):
            distance = modifier.get_distance().gen_physical_value()

            if location == "ahead_of":
                wp_lists = init_wp.next(distance)
            else:
                wp_lists = init_wp.previous(distance)
            if wp_lists:
                init_wp = wp_lists[0]
        else:
            raise KeyError(f"wrong location = {location}")

    if init_wp:
        actor = CarlaDataProvider.get_actor_by_name(npc_name)
        npc_car_visible = ActorTransformSetter(actor, init_wp.transform)
        father_tree.add_child(npc_car_visible)

        car_config = config.get_car_config(npc_name)
        car_config.set_arg({"init_transform": init_wp.transform})
        LOG_WARNING(
            f"{npc_name} car init position will be set to {init_wp.transform.location},roadid = {init_wp.road_id}, laneid={init_wp.lane_id}, s={init_wp.s}"
        )

    # end
    end_group = [m for m in modifiers if m.get_trigger_point() == "end"]

    end_wp = None
    end_lane_wp = None

    for modifier in end_group:
        npc_name = modifier.get_actor_name()

        relative_car_name, location = modifier.get_refer_car()
        relative_car_conf = config.get_car_config(relative_car_name)
        relative_car_location = relative_car_conf.get_transform().location
        LOG_WARNING(f"{relative_car_name} pos = {relative_car_location}")

        relative_car_wp = CarlaDataProvider.get_map().get_waypoint(
            relative_car_location
        )
        relative_car_speed = relative_car_conf.get_arg("target_speed")

        distance_will_drive = relative_car_speed * float(duration)
        LOG_WARNING(f"{relative_car_name} drive distance = {distance_will_drive}")

        end_position = relative_car_wp.next(distance_will_drive)
        if end_position is None or len(end_position) == 0:
            raise RuntimeError("the road is not long enough")

        end_position = end_position[0]

        if location in ('ahead_of', 'behind'):
            # End position constraint
            distance = modifier.get_distance().gen_physical_value()
            if location == "ahead_of":
                wp_lists = end_position.next(distance)
            else:
                wp_lists = end_position.previous(distance)
            if wp_lists:
                end_wp = wp_lists[0]
        elif location in ('left_of', 'right_of', 'same_as'):
            # Lane restraint at the end
            if location == "left_of":
                temp_wp = relative_car_wp.get_left_lane()
            elif location == "right_of":
                temp_wp = relative_car_wp.get_right_lane()
            elif location == "same_as":
                temp_wp = relative_car_wp
            else:
                LOG_INFO("lane spec is error")

            end_lane_wp = temp_wp
        else:
            raise RuntimeError("relative position is igeal")

    if end_wp:
        current_car_conf = config.get_car_config(npc_name)
        current_car_transform = current_car_conf.get_arg("init_transform")

        # Get the global route planner, used to calculate the route
        grp = GlobalRoutePlanner(CarlaDataProvider.get_world().get_map(), 0.5)
        # grp.setup()

        distance = calculate_distance(
            current_car_transform.location, end_wp.transform.location, grp
        )

        car_need_speed = distance / float(duration)

        current_car_conf.set_arg({"desired_speed": car_need_speed})
        LOG_WARNING(
            f"{npc_name} car desired speed will be set to {car_need_speed * 3.6} km/h"
        )

        car_actor = CarlaDataProvider.get_actor_by_name(npc_name)
        car_driving = WaypointFollower(car_actor, car_need_speed)
        # car_driving.set_duration(duration)
        father_tree.add_child(car_driving)

    if end_lane_wp:
        current_car_conf = config.get_car_config(npc_name)
        current_car_transform = current_car_conf.get_arg("init_transform")
        car_lane_wp = CarlaDataProvider.get_map().get_waypoint(
            current_car_transform.location
        )

        direction = None
        if end_lane_wp and car_lane_wp:
            end_lane_id = end_lane_wp.lane_id
            end_lane = None
            if end_lane_id == car_lane_wp.get_left_lane().lane_id:
                direction = "left"
                end_lane = car_lane_wp.get_left_lane()
            elif end_lane_id == car_lane_wp.get_right_lane().lane_id:
                direction = "right"
                end_lane = car_lane_wp.get_right_lane()
            else:
                print("no need change lane")

            car_actor = CarlaDataProvider.get_actor_by_name(npc_name)

            lane_change = LaneChange(
                car_actor,
                speed=None,
                direction=direction,
                distance_same_lane=5,
                distance_other_lane=10,
            )
            # lane_change.set_duration(duration)

            if end_lane:
                # After lane change, the car needs to modify its lane information.
                # Here, there should be a special variable to save the current transform
                car_config = config.get_car_config(npc_name)
                car_config.set_arg({"init_transform": end_lane.transform})

            continue_drive = WaypointFollower(car_actor)
            # car_driving.set_duration(duration)
            father_tree.add_child(lane_change)
            father_tree.add_child(continue_drive)


class OSC2Scenario(BasicScenario):
    """
    Implementation of the osc2 Scenario
    """

    def __init__(
        self,
        world,
        ego_vehicles,
        config: OSC2ScenarioConfiguration,
        osc2_file,
        debug_mode=False,
        criteria_enable=True,
        timeout=300,
    ):
        """
        Setup all relevant parameters and create scenario
        """
        self.config = config
        self.route = None
        self.osc2_file = osc2_file
        self.ast_tree = OSC2Helper.gen_osc2_ast(self.osc2_file)
        # Timeout of scenario in seconds
        self.timeout = timeout
        self.all_duration = float()

        self.other_actors = None

        self.behavior = None

        # Check whether the access is available
        self.visit_power = False
        self.scenario_declaration = config.scenario_declaration
        self.struct_declaration = config.struct_declaration
        # Use struct_parameters to store parameters of type struct, so that we can recognize it in keep constraint
        self.struct_parameters = {}

        super(OSC2Scenario, self).__init__(
            "OSC2Scenario",
            ego_vehicles=ego_vehicles,
            config=config,
            world=world,
            debug_mode=debug_mode,
            terminate_on_failure=False,
            criteria_enable=criteria_enable,
        )

    def set_behavior_tree(self, behavior):
        self.behavior = behavior

    class BehaviorInit(ASTVisitor):
        def __init__(self, config_instance) -> None:
            super().__init__()
            self.father_ins = config_instance
            self.root_behavior = None
            self.__cur_behavior = None
            self.__parent_behavior = {}
            self.__duration = 1000000000.0

        def get_behavior_tree(self):
            return self.root_behavior

        def visit_scenario_declaration(self, node: ast_node.ScenarioDeclaration):
            scenario_name = node.qualified_behavior_name

            if scenario_name != "top" and not self.father_ins.visit_power:
                return

            if scenario_name == "top" and self.father_ins.visit_power:
                return

            for child in node.get_children():
                if isinstance(child, ast_node.DoDirective):
                    self.visit_do_directive(child)
                elif isinstance(child, ast_node.ModifierInvocation):
                    self.visit_modifier_invocation(child)
                elif isinstance(child, ast_node.ParameterDeclaration):
                    self.visit_parameter_declaration(child)
                elif isinstance(child, ast_node.KeepConstraintDeclaration):
                    self.visit_keep_constraint_declaration(child)

        def visit_do_directive(self, node: ast_node.DoDirective):
            self.visit_children(node)

        def bool_result(self, option):
            # wait(x < y) @drive_distance Handling of Boolean expressions x < y
            expression_value = re.split("\W+", option)
            symbol = re.search("\W+", option).group()
            if symbol == "<":
                symbol = operator.lt
            elif symbol == ">":
                symbol = operator.gt
            # if len(expression_value) == 2:
            #     x = variables.Variable.get_arg(expression_value[0])
            #     y = variables.Variable.get_arg(expression_value[1])
            x = expression_value[0]
            y = expression_value[1]
            if "ego" in x:
                actor_name = "ego_vehicle"
                actor_ego = CarlaDataProvider.get_actor_by_name(actor_name)
            if "npc" in y:
                actor_name = "npc"
                actor_npc = CarlaDataProvider.get_actor_by_name(actor_name)
            return actor_ego, actor_npc, symbol

        def visit_do_member(self, node: ast_node.DoMember):
            self.__duration = 1000000000.0
            composition_operator = node.composition_operator
            sub_node = None
            if composition_operator in ["serial", "parallel", "one_of"]:
                if composition_operator == "serial":
                    self.__cur_behavior = py_trees.composites.Sequence(
                        policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL,
                        name="serial",
                    )
                elif composition_operator == "parallel":
                    self.__cur_behavior = py_trees.composites.Parallel(
                        policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL,
                        name="parallel",
                    )
                elif composition_operator == "one_of":
                    self.__cur_behavior = py_trees.composites.Sequence(
                        policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL,
                        name="one_of",
                    )
                    do_member_list = []
                    for child in node.get_children():
                        if isinstance(child, ast_node.DoMember):
                            do_member_list.append(child)
                    sub_node = random.choice(do_member_list)
            else:
                raise NotImplementedError(
                    f"no supported scenario operator {composition_operator}"
                )

            if self.root_behavior is None:
                self.root_behavior = self.__cur_behavior
                self.__parent_behavior[node] = self.__cur_behavior
            elif (
                self.root_behavior is not None
                and self.__parent_behavior.get(node) is None
            ):
                self.__parent_behavior[node] = self.root_behavior
                parent = self.__parent_behavior[node]
                parent.add_child(self.__cur_behavior)
            else:
                parent = self.__parent_behavior[node]
                parent.add_child(self.__cur_behavior)

            for child in node.get_children():
                if not isinstance(child, ast_node.AST):
                    continue

                if isinstance(child, (ast_node.DoMember, ast_node.EmitDirective, ast_node.WaitDirective)):
                    self.__parent_behavior[child] = self.__cur_behavior

            if sub_node is None:
                for child in node.get_children():
                    if not isinstance(child, ast_node.AST):
                        continue

                    if isinstance(child, ast_node.DoMember):
                        self.visit_do_member(child)
                    elif isinstance(child, ast_node.NamedArgument):
                        named_arg = self.visit_named_argument(child)
                        if named_arg[0] == "duration":
                            if isinstance(named_arg[1], Physical):
                                self.__duration = named_arg[1].gen_physical_value()
                            else:
                                print(
                                    "[Error] 'duration' parameter must be 'Physical' type"
                                )
                                sys.exit(1)
                    elif isinstance(child, ast_node.BehaviorInvocation):
                        self.visit_behavior_invocation(child)
                    elif isinstance(child, ast_node.WaitDirective):
                        self.visit_wait_directive(child)
                    elif isinstance(child, ast_node.EmitDirective):
                        self.visit_emit_directive(child)
                    elif isinstance(child, ast_node.CallDirective):
                        self.visit_call_directive(child)
                    else:
                        raise NotImplementedError(f"no implentment AST node {child}")
            else:
                if isinstance(sub_node, ast_node.DoMember):
                    self.visit_do_member(sub_node)
                else:
                    raise NotImplementedError("no supported ast node")

            if re.match("\d", str(self.__duration)) and self.__duration != math.inf:
                self.father_ins.all_duration += int(self.__duration)

        def visit_wait_directive(self, node: ast_node.WaitDirective):
            behaviors = py_trees.composites.Sequence(
                policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL, name="wait"
            )
            subbehavior = py_trees.composites.Sequence(
                policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL, name="behavior"
            )

            if node.get_child_count() == 1 and isinstance(
                node.get_child(0), ast_node.EventCondition
            ):
                elapsed_condition = self.visit_event_condition(node.get_child(0))
                self.__duration = elapsed_condition.gen_physical_value()
                print(elapsed_condition, self.__duration)
                self.father_ins.all_duration += int(self.__duration)
                waitTriggerer = TimeOfWaitComparison(self.__duration)
                waitTriggerer = oneshot_with_check(
                    variable_name="wait_time", behaviour=waitTriggerer
                )
                subbehavior.add_child(waitTriggerer)
                behaviors.add_child(subbehavior)
                parent = self.__parent_behavior[node]
                parent.add_child(behaviors)
                return

            bool_condition = ""
            for child in node.get_children():
                if not isinstance(child, ast_node.AST):
                    continue
                if isinstance(child, ast_node.EventReference):
                    event_declaration_node, event_name = self.visit_event_reference(
                        child
                    )
                elif isinstance(child, ast_node.EventFieldDecl):
                    pass
                elif isinstance(child, ast_node.EventCondition):
                    # string bool_condition = 'x<y'
                    bool_condition = self.visit_event_condition(child)
                else:
                    raise NotImplementedError(f"no implentment AST node {child}")
            actor_ego, actor_npc, symbol = self.bool_result(bool_condition)
            arguments = self.visit_children(event_declaration_node)
            other_car = arguments[0][1]
            distance = float(arguments[1][1].gen_physical_value())
            ret = getattr(self.father_ins.event, event_name)(other_car, distance)
            ret = oneshot_with_check(variable_name="wait_condition", behaviour=ret)

            wait_triggerer = IfTriggerer(actor_ego, actor_npc, symbol)
            wait_triggerer = oneshot_with_check(
                variable_name="wait", behaviour=wait_triggerer
            )
            subbehavior.add_child(wait_triggerer)
            subbehavior.add_child(ret)
            behaviors.add_child(subbehavior)

            parent = self.__parent_behavior[node]
            parent.add_child(behaviors)

        def visit_emit_directive(self, node: ast_node.EmitDirective):
            behaviors = py_trees.composites.Sequence(
                policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL, name="emit"
            )
            function_name = node.event_name
            arguments = self.visit_children(node)
            actor = arguments[0][1]
            distance = float(arguments[1][1].gen_physical_value())
            ret = getattr(self.father_ins.event, function_name)(actor, distance)
            ret = oneshot_with_check(variable_name="emit_condition", behaviour=ret)
            behaviors.add_child(ret)

            parent = self.__parent_behavior[node]
            parent.add_child(behaviors)

        def visit_behavior_invocation(self, node: ast_node.BehaviorInvocation):
            actor = node.actor
            behavior_name = node.behavior_name

            behavior_invocation_name = None
            if actor != None:
                behavior_invocation_name = actor + "." + behavior_name
            else:
                behavior_invocation_name = behavior_name

            if (
                self.father_ins.scenario_declaration.get(behavior_invocation_name)
                is not None
            ):
                self.father_ins.visit_power = True
                scenario_declaration_node = copy.deepcopy(
                    node.get_scope().declaration_address
                )
                # scenario_declaration_node = self.father_ins.scenario_declaration.get(behavior_invocation_name)
                scenario_declaration_node_scope = scenario_declaration_node.get_scope()
                arguments = self.visit_children(node)
                # Stores the value of the argument before the invoked scenario was overwritten， a: time=None
                # keyword_args = {}
                if isinstance(arguments, List):
                    for arg in arguments:
                        if isinstance(arg, Tuple):
                            scope = scenario_declaration_node_scope.resolve(arg[0])
                            # keyword_args[arg[0]] = scope.value
                            scope.value = arg[1]
                elif isinstance(arguments, Tuple):
                    scope = scenario_declaration_node_scope.resolve(arguments[0])
                    # keyword_args[arguments[0]] = scope.value
                    scope.value = arguments[1]
                self.visit_scenario_declaration(scenario_declaration_node)
                # Restores the value of the argument before the called scene was overwritten
                # for (name,value) in keyword_args.items():
                #     scope = scenario_declaration_node_scope.resolve(name)
                #     scope.value = value
                del scenario_declaration_node
                return

            behavior = py_trees.composites.Parallel(
                policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE,
                name=behavior_invocation_name
                + " duration="
                + str(int(self.__duration)),
            )

            # Create node for timeout
            timeout = TimeOut(
                self.__duration, name="duration=" + str(int(self.__duration))
            )
            behavior.add_child(timeout)

            actor_drive = py_trees.composites.Sequence(
                policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL,
                name=behavior_invocation_name,
            )

            modifier_invocation_no_occur = True

            location_modifiers = []
            speed_modifiers = []

            children = node.get_children()
            for child in children:
                if isinstance(child, ast_node.NamedArgument):
                    named_arg = self.visit_named_argument(child)
                    if named_arg[0] == "duration" and isinstance(
                        named_arg[1], Physical
                    ):
                        self.__duration = named_arg[1].gen_physical_value()
                    elif named_arg[0] == "duration":
                        print("[Error] 'duration' parameter must be 'Physical' type")
                        # sys.exit(1)
                elif isinstance(child, ast_node.ModifierInvocation):
                    modifier_invocation_no_occur = False
                    modifier = self.visit_modifier_invocation(child)
                    modifier_name = modifier[0]
                    arguments = modifier[1]

                    if modifier_name == "speed":
                        modifier_ins = SpeedModifier(actor, modifier_name)
                        keyword_args = {}
                        if isinstance(arguments, list):
                            arguments = OSC2Helper.flat_list(arguments)

                            for arg in arguments:
                                if isinstance(arg, tuple):
                                    keyword_args[arg[0]] = arg[1]
                                elif isinstance(arg, Physical):
                                    keyword_args["speed"] = arg
                        elif isinstance(arguments, tuple):
                            keyword_args[arguments[0]] = arguments[1]
                        elif isinstance(arguments, Physical):
                            keyword_args["speed"] = arguments
                        else:
                            raise NotImplementedError(
                                f"no implentment argument of {modifier_name}"
                            )

                        modifier_ins.set_args(keyword_args)

                        speed_modifiers.append(modifier_ins)

                    elif modifier_name == "position":
                        modifier_ins = PositionModifier(actor, modifier_name)
                        keyword_args = {}

                        keyword_args = {}
                        if isinstance(arguments, list):
                            arguments = OSC2Helper.flat_list(arguments)
                            for arg in arguments:
                                if isinstance(arg, tuple):
                                    keyword_args[arg[0]] = arg[1]
                                elif isinstance(arg, Physical):
                                    keyword_args["distance"] = arg
                        elif isinstance(arguments, tuple):
                            keyword_args[arguments[0]] = arguments[1]
                        elif isinstance(arg, Physical):
                            keyword_args["distance"] = arguments
                        else:
                            raise NotImplementedError(
                                f"no implentment argument of {modifier_name}"
                            )

                        modifier_ins.set_args(keyword_args)

                        location_modifiers.append(modifier_ins)

                    elif modifier_name == "lane":
                        modifier_ins = LaneModifier(actor, modifier_name)

                        keyword_args = {}
                        if isinstance(arguments, List):
                            arguments = OSC2Helper.flat_list(arguments)
                            for arg in arguments:
                                if isinstance(arg, Tuple):
                                    keyword_args[arg[0]] = arg[1]
                                else:
                                    keyword_args["lane"] = str(arg)
                        elif isinstance(arguments, Tuple):
                            keyword_args[arguments[0]] = arguments[1]
                        else:
                            keyword_args["lane"] = str(arguments)
                        modifier_ins.set_args(keyword_args)

                        location_modifiers.append(modifier_ins)

                    elif modifier_name == "acceleration":
                        modifier_ins = AccelerationModifier(actor, modifier_name)

                        keyword_args = {}
                        if isinstance(arguments, List):
                            arguments = OSC2Helper.flat_list(arguments)
                            for arg in arguments:
                                if isinstance(arg, Tuple):
                                    keyword_args[arg[0]] = arg[1]
                                else:
                                    keyword_args["acceleration"] = arg
                        elif isinstance(arguments, Tuple):
                            keyword_args[arguments[0]] = arguments[1]
                        else:
                            keyword_args["acceleration"] = arguments
                        modifier_ins.set_args(keyword_args)
                        speed_modifiers.append(modifier_ins)

                    elif modifier_name == "keep_lane":
                        actor_object = CarlaDataProvider.get_actor_by_name(actor)
                        car_driving = WaypointFollower(actor_object)
                        actor_drive.add_child(car_driving)

                        behavior.add_child(actor_drive)
                        # self.__cur_behavior.add_child(behavior)
                        print("Target keep lane.")

                    elif modifier_name == "change_speed":
                        # change_speed([speed: ]<speed>)
                        modifier_ins = ChangeSpeedModifier(actor, modifier_name)

                        keyword_args = {}
                        if isinstance(arguments, Tuple):
                            keyword_args[arguments[0]] = arguments[1]
                        elif isinstance(arguments, Physical):
                            keyword_args["desired_speed"] = arguments
                        else:
                            return f"Needed 1 arguments, but given {len(arguments)}arguments."

                        modifier_ins.set_args(keyword_args)

                        speed_modifiers.append(modifier_ins)

                    elif modifier_name == "change_lane":
                        modifier_ins = ChangeLaneModifier(actor, modifier_name)

                        keyword_args = {}
                        count = 0
                        for _, _ in enumerate(arguments):
                            count += 1
                        if count == 1:
                            keyword_args["side"] = arguments
                        elif count == 2:
                            if isinstance(arguments[0], Tuple):
                                keyword_args[arguments[0][0]] = str(arguments[0][1])
                            else:
                                keyword_args["lane_changes"] = str(arguments[0])
                            if isinstance(arguments[1], Tuple):
                                keyword_args[arguments[1][0]] = arguments[1][1]
                            else:
                                keyword_args["side"] = arguments[1]
                        else:
                            return f"Needed 2 arguments, but given {len(arguments)}arguments."

                        modifier_ins.set_args(keyword_args)

                        location_modifiers.append(modifier_ins)
                    else:
                        raise NotImplementedError(
                            f"no implentment function: {modifier_name}"
                        )

            if modifier_invocation_no_occur:
                car_actor = CarlaDataProvider.get_actor_by_name(actor)
                car_driving = WaypointFollower(car_actor)
                actor_drive.add_child(car_driving)
                behavior.add_child(actor_drive)
                self.__cur_behavior.add_child(behavior)
                return

            process_location_modifier(
                self.father_ins.config, location_modifiers, self.__duration, actor_drive
            )
            process_speed_modifier(
                self.father_ins.config,
                speed_modifiers,
                self.__duration,
                self.father_ins.all_duration,
                actor_drive,
            )

            behavior.add_child(actor_drive)
            self.__cur_behavior.add_child(behavior)

        def visit_modifier_invocation(self, node: ast_node.ModifierInvocation):
            # actor = node.actor
            modifier_name = node.modifier_name
            LOG_INFO(f"modifier invocation name {node.modifier_name}")
            arguments = self.visit_children(node)
            line, column = node.get_loc()
            # retrieval_name = modifier_name + para_type_str_sequence(config=self.father_ins.config,
            # arguments=arguments, line=line, column=column, node=node)
            retrieval_name = modifier_name
            method_scope = node.get_scope().resolve(retrieval_name)
            if (
                method_scope is None
                and modifier_name not in dir(self.father_ins.config.path)
                and modifier_name
                not in (
                    "speed",
                    "lane",
                    "position",
                    "acceleration",
                    "keep_lane",
                    "change_speed",
                    "change_lane",
                )
            ):
                line, column = node.get_loc()
                LOG_ERROR(
                    "Not Find " + modifier_name + " Method Declaration",
                    token=None,
                    line=line,
                    column=column,
                )
            if isinstance(method_scope, MethodSymbol):
                method_declaration_node = copy.deepcopy(
                    method_scope.declaration_address
                )
                method_scope = method_declaration_node.get_scope()
                if isinstance(arguments, List):
                    for arg in arguments:
                        if isinstance(arg, Tuple):
                            scope = method_scope.resolve(arg[0])
                            scope.value = arg[1]
                elif isinstance(arguments, Tuple):
                    scope = method_scope.resolve(arguments[0])
                    scope.value = arguments[1]
                method_value = None
                for child in method_declaration_node.get_children():
                    if isinstance(child, ast_node.MethodBody):
                        method_value = self.visit_method_body(child)
                del method_declaration_node
                if method_value is not None:
                    return method_value
                return
            else:
                pass
            arguments = self.visit_children(node)
            return modifier_name, arguments

        def visit_event_reference(self, node: ast_node.EventReference):
            return (
                copy.deepcopy(
                    node.get_scope().resolve(node.event_path).declaration_address
                ),
                node.event_path,
            )
            # return node.event_path

        def visit_event_field_declaration(self, node: ast_node.EventFieldDecl):
            return super().visit_event_field_declaration(node)

        def visit_event_condition(self, node: ast_node.EventCondition):
            expression = ""
            for child in node.get_children():
                if isinstance(child, ast_node.RelationExpression):
                    flat_arguments = self.visit_relation_expression(child)
                    temp_stack = []
                    for ex in flat_arguments:
                        if ex in RelationalOperator.values():
                            right = temp_stack.pop()
                            left = temp_stack.pop()
                            expression = left + ex + str(right)
                            temp_stack.append(expression)
                        elif ex == "in":
                            right = temp_stack.pop()
                            left = temp_stack.pop()
                            innum = temp_stack.pop()
                            expression = (
                                innum + " " + ex + " [" + left + ", " + right + "]"
                            )
                            temp_stack.append(expression)
                        else:
                            temp_stack.append(ex)
                    expression = temp_stack.pop()
                elif isinstance(child, ast_node.LogicalExpression):
                    expression = self.visit_logical_expression(child)
                elif isinstance(child, ast_node.ElapsedExpression):
                    expression = self.visit_elapsed_expression(child)
                else:
                    pass
            return expression

        def visit_relation_expression(self, node: ast_node.RelationExpression):
            arguments = [self.visit_children(node), node.operator]
            flat_arguments = OSC2Helper.flat_list(arguments)
            return flat_arguments

        def visit_logical_expression(self, node: ast_node.LogicalExpression):
            arguments = [self.visit_children(node), node.operator]
            flat_arguments = OSC2Helper.flat_list(arguments)
            temp_stack = []
            for ex in flat_arguments:
                if ex in ('and', 'or', '=>'):
                    expression = ""
                    length = len(temp_stack) - 1
                    for num in temp_stack:
                        if length > 0:
                            expression = expression + num + " " + ex + " "
                            length = length - 1
                        else:
                            expression = expression + num
                    temp_stack.clear()
                    temp_stack.append(expression)
                elif ex == "not":
                    num = temp_stack.pop()
                    expression = ex + " " + num
                    temp_stack.append(expression)
                else:
                    temp_stack.append(ex)
            logical_expression = temp_stack.pop()
            # return [self.visit_children(node), node.operator]
            return logical_expression

        def visit_elapsed_expression(self, node: ast_node.ElapsedExpression):
            child = node.get_child(0)
            if isinstance(child, ast_node.PhysicalLiteral):
                return self.visit_physical_literal(child)
            elif isinstance(child, ast_node.RangeExpression):
                return self.visit_range_expression(child)
            else:
                return None

        def visit_binary_expression(self, node: ast_node.BinaryExpression):
            arguments = [self.visit_children(node), node.operator]
            flat_arguments = OSC2Helper.flat_list(arguments)
            LOG_INFO(f"{flat_arguments}")
            temp_stack = []
            for ex in flat_arguments:
                if ex in ('+', '-', '*', '/', '%'):
                    right = temp_stack.pop()
                    left = temp_stack.pop()
                    # expression = left + ' ' + ex + ' ' + right
                    if ex == "+":
                        expression = left + right
                    elif ex == "-":
                        expression = left - right
                    elif ex == "*":
                        expression = left * right
                    elif ex == "/":
                        expression = left / right
                    elif ex == "%":
                        expression = left % right
                    else:
                        LOG_INFO(f"undefined Binary Expression operator: {ex}")
                    temp_stack.append(expression)
                else:
                    temp_stack.append(ex)
            binary_expression = temp_stack.pop()
            LOG_INFO(f"Relation Expression Value: {binary_expression}")
            # return [self.visit_children(node), node.operator]
            return binary_expression

        def visit_named_argument(self, node: ast_node.NamedArgument):
            return node.argument_name, self.visit_children(node)

        def visit_positional_argument(self, node: ast_node.PositionalArgument):
            return self.visit_children(node)

        def visit_range_expression(self, node: ast_node.RangeExpression):
            start, end = self.visit_children(node)
            if type(start) != type(end):
                print("[Error] different types between start and end of the range")
                sys.exit(1)

            start_num = None
            end_num = None
            start_unit = None
            end_unit = None
            unit_name = None

            if isinstance(start, Physical):
                start_num = start.num
                end_num = end.num

                start_unit = start.unit
                end_unit = end.unit
            else:
                start_num = start
                end_num = end

            if start_unit is not None and end_unit is not None:
                if start_unit == end_unit:
                    unit_name = start_unit
                else:
                    print("[Error] wrong unit in the range")
                    sys.exit(1)

            if start_num >= end_num:
                print("[Error] wrong start and end in the range")
                sys.exit(1)

            var_range = Range(start_num, end_num)

            if unit_name:
                return Physical(var_range, unit_name)
            else:
                return var_range

        def visit_physical_literal(self, node: ast_node.PhysicalLiteral):
            return Physical(
                self.visit_children(node),
                self.father_ins.config.unit_dict[node.unit_name],
            )

        def visit_integer_literal(self, node: ast_node.IntegerLiteral):
            return int(node.value)

        def visit_float_literal(self, node: ast_node.FloatLiteral):
            return float(node.value)

        def visit_bool_literal(self, node: ast_node.BoolLiteral):
            return node.value == "true"

        def visit_string_literal(self, node: ast_node.StringLiteral):
            return node.value

        def visit_identifier(self, node: ast_node.Identifier):
            return node.name

        def visit_identifier_reference(self, node: ast_node.IdentifierReference):
            para_name = node.name
            para_type = None
            para_value = None
            if node.get_scope() is not None:
                if not hasattr(node.get_scope(), "type"):
                    return para_name
                para_type = node.get_scope().type
                symbol = node.get_scope()
                last_value = None
                cur_value = node.get_scope().value
                while last_value != cur_value and symbol.resolve(cur_value) is not None:
                    symbol = symbol.resolve(cur_value)
                    last_value = cur_value
                    cur_value = symbol.value
                if cur_value is None:
                    return symbol.name
                else:
                    para_value = cur_value
            if para_value is not None:
                if isinstance(para_value, (Physical, float, int)):
                    return para_value
                para_value = para_value.strip('"')
                if re.fullmatch("(^[-]?[0-9]+(\.[0-9]+)?)\s*(\w+)", para_value):
                    # Regular expression ^[-]?[0-9]+(\.[0-9]+)? matching float
                    # para_value_num = re.findall('^[-]?[0-9]+(\.[0-9]+)?', para_value)[0]
                    patter = re.compile("(^[-]?[0-9]+[\.[0-9]+]?)\s*(\w+)")
                    para_value_num, para_value_unit = patter.match(para_value).groups()
                    if para_value_num.count(".") == 1:
                        return Physical(
                            float(para_value_num),
                            self.father_ins.config.unit_dict[para_value_unit],
                        )
                    else:
                        return Physical(
                            int(para_value_num),
                            self.father_ins.config.unit_dict[para_value_unit],
                        )
                elif para_type == "int":
                    return int(para_value)
                elif para_type == "uint":
                    return int(para_value)
                elif para_type == "float":
                    return float(para_value)
                elif para_type == "bool":
                    return para_value == "true"
                elif para_type == "string":
                    return para_value
                else:
                    return para_value
            else:
                return para_name

        def visit_parameter_declaration(self, node: ast_node.ParameterDeclaration):
            para_name = node.field_name
            para_type = node.field_type
            para_value = None
            for child in node.get_children():
                if isinstance(child, ast_node.FunctionApplicationExpression):
                    para_value = self.visit_function_application_expression(child)
            if para_value is not None:
                node.get_scope().value = para_value

            # Save variables of type struct for later access
            if para_type in self.father_ins.struct_declaration:
                self.father_ins.struct_parameters.update(
                    {para_name[0]: self.father_ins.struct_declaration[para_type]}
                )

        def visit_method_declaration(self, node: ast_node.MethodDeclaration):
            pass

        def visit_argument(self, node: ast_node.Argument):
            return node.argument_name, self.visit_children(node)

        def visit_method_body(self, node: ast_node.MethodBody):
            type = node.type
            method_value = None
            if type == "external":
                external_list = []
                for child in node.get_children():
                    if isinstance(child, ast_node.PositionalArgument):
                        line, column = node.get_loc()
                        LOG_ERROR(
                            "not support external format.!",
                            token=None,
                            line=line,
                            column=column,
                        )
                    elif isinstance(child, ast_node.NamedArgument):
                        name, value = self.visit_named_argument(child)
                        external_list.append((name, value))

                exec_context = ""
                module_name = None
                for elem in external_list:
                    if "module" == elem[0]:
                        exec_context += "import " + str(elem[1]) + "\n"
                        module_name = str(elem[1])
                    elif "name" == elem[0]:
                        exec_context += "ret = "
                        if module_name is not None:
                            exec_context += module_name + "." + str(elem[1]) + "("
                        else:
                            exec_context += str(elem[1]) + "("
                    else:
                        exec_context += str(elem[1])
                exec_context += ")\n"

                try:
                    exec_data = {}
                    exec(exec_context, globals(), exec_data)
                    method_value = exec_data["ret"]
                except Exception:
                    line, column = node.get_loc()
                    LOG_ERROR(
                        "not support external format.!",
                        token=None,
                        line=line,
                        column=column,
                    )

                return method_value
            else:
                for child in node.get_children():
                    if isinstance(child, ast_node.BinaryExpression):
                        method_value = self.visit_binary_expression(child)
            if method_value is not None:
                return method_value
            return

        def visit_function_application_expression(
            self, node: ast_node.FunctionApplicationExpression
        ):
            LOG_INFO("visit function application expression!")
            LOG_INFO("func name:" + node.func_name)

            arguments = OSC2Helper.flat_list(self.visit_children(node))
            line, column = node.get_loc()
            # retrieval_name = para_type_str_sequence(config=self.father_ins.config, arguments=arguments, line=line, column=column, node=node)
            retrieval_name = arguments[0].split(".")[-1]
            method_scope = node.get_scope().resolve(retrieval_name)

            method_name = arguments[0]
            if method_scope is None:
                LOG_ERROR(
                    "Not Find " + method_name + " Method Declaration",
                    token=None,
                    line=line,
                    column=column,
                )
            para_value = None
            if isinstance(method_scope, MethodSymbol):
                method_declaration_node = copy.deepcopy(
                    method_scope.declaration_address
                )
                method_scope = method_declaration_node.get_scope()
                if isinstance(arguments, List):
                    for arg in arguments:
                        if isinstance(arg, Tuple):
                            scope = method_scope.resolve(arg[0])
                            scope.value = arg[1]
                elif isinstance(arguments, Tuple):
                    scope = method_scope.resolve(arguments[0])
                    scope.value = arguments[1]
                para_value = None
                for child in method_declaration_node.get_children():
                    if isinstance(child, ast_node.MethodBody):
                        para_value = self.visit_method_body(child)
                        break
                del method_declaration_node
                if para_value is not None:
                    return para_value
                return para_value
            else:
                pass

        def visit_keep_constraint_declaration(
            self, node: ast_node.KeepConstraintDeclaration
        ):
            arguments = self.visit_children(node)
            retrieval_name = arguments[0]

            # Struct parameter or actor parameter contains '.'
            if "." in retrieval_name:
                layered_names = retrieval_name.split(".")
                prefix = layered_names[0]
                suffix = layered_names[1:]
                if prefix == "it":
                    pass
                elif prefix in self.father_ins.struct_parameters:
                    # param_name is the name of the struct variable, and param_scope is a ParameterSymbol
                    param_scope = node.get_scope().resolve(prefix)
                    self._build_struct_tree(param_scope)
                    self._visit_struct_tree(param_scope, suffix, 0, arguments[1])
            param_scope = node.get_scope().resolve(retrieval_name)
            if param_scope is not None and isinstance(param_scope, ParameterSymbol):
                if arguments[2] == RelationalOperator.EQUALITY.value:
                    param_scope.value = arguments[1]
                elif arguments[2] == RelationalOperator.INEQUALITY.value:
                    pass
                elif arguments[2] == RelationalOperator.LESS_THAN.value:
                    pass
                elif arguments[2] == RelationalOperator.LESS_OR_EQUAL.value:
                    pass
                elif arguments[2] == RelationalOperator.GREATER_THAN.value:
                    pass
                elif arguments[2] == RelationalOperator.GREATER_OR_EQUAL.value:
                    pass
                elif arguments[2] == RelationalOperator.MEMBERSHIP.value:
                    pass

        # For variables of struct type, it is necessary to construct its struct variable tree in the symbol table
        # The struct variable tree is a subtree of the symbol tree
        def _build_struct_tree(self, param_symbol: ParameterSymbol):
            if param_symbol.value is None:
                param_symbol.value = copy.deepcopy(
                    self.father_ins.struct_parameters[param_symbol.name]
                ).get_scope()
            for key in param_symbol.value.symbols:
                child_symbol = param_symbol.value.symbols[key]
                if isinstance(child_symbol, ParameterSymbol):
                    # If the child parameter is of struct type, the current method is called recursively
                    if child_symbol.type in self.father_ins.struct_declaration:
                        self._build_struct_tree(child_symbol)

        # visit struct variable tree and assign value
        def _visit_struct_tree(
            self, root: ParameterSymbol, suffix: list, index: int, value
        ):
            if root.type not in self.father_ins.struct_declaration:
                root.value = value
                return
            if index >= len(suffix):
                return
            to_visit_param = suffix[index]
            child_symbols = root.value.symbols
            if to_visit_param not in child_symbols:
                return
            self._visit_struct_tree(
                child_symbols[to_visit_param], suffix, index + 1, value
            )

    def _create_behavior(self):
        """
        Basic behavior do nothing, i.e. Idle
        """
        behavior_builder = self.BehaviorInit(self)
        behavior_builder.visit(self.ast_tree)

        behavior_tree = behavior_builder.get_behavior_tree()
        self.set_behavior_tree(behavior_tree)

        # py_trees.display.render_dot_tree(behavior_tree)

        return self.behavior

    def _create_test_criteria(self):
        """
        A list of all test criteria is created, which is later used in the parallel behavior tree.
        """
        criteria = []

        collision_criterion = CollisionTest(self.ego_vehicles[0])

        criteria.append(collision_criterion)

        return criteria

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()
