
"""
Parse the OSC2 scenario description file, configure parameters based on type and keep constraints, 
generate relevant type objects in the standard library, and set parameters

"""
import sys
from typing import List, Tuple

import carla

import srunner.osc2_stdlib.misc_object as misc
import srunner.osc2_stdlib.variables as variable
import srunner.osc2_stdlib.vehicle as vehicles
from srunner.osc2.ast_manager import ast_node
from srunner.osc2.ast_manager.ast_vistor import ASTVisitor
from srunner.osc2_dm.physical_object import *
from srunner.osc2_dm.physical_types import Physical, Range

from srunner.osc2_stdlib.path import Path

# pylint: disable=line-too-long
from srunner.scenarioconfigs.scenario_configuration import ScenarioConfiguration
'''
Parses the osc2 scenario description file, generates type objects in the standard 
library based on the type and keep constraint parameters, and sets parameters
'''

# pylint: enable=line-too-long
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

# OSC2
from srunner.tools.osc2_helper import OSC2Helper

vehicle_type = ["Car", "Model3", "Mkz2017", "Carlacola", "Rubicon"]


def flat_list(list_of_lists):
    if len(list_of_lists) == 0:
        return list_of_lists

    if isinstance(list_of_lists[0], list):
        return flat_list(list_of_lists[0]) + flat_list(list_of_lists[1:])

    return list_of_lists[:1] + flat_list(list_of_lists[1:])


class OSC2ScenarioConfiguration(ScenarioConfiguration):
    def __init__(self, filename, client):
        super(OSC2ScenarioConfiguration, self).__init__()
        
        self.name = self.filename = filename
        self.ast_tree = OSC2Helper.gen_osc2_ast(self.filename)

        self.client = client

        self.path = Path
        self.other_actors = []
        self.ego_vehicles = []
        self.all_actors = {}
        self.variables = {}
        self.unit_dict = {}
        self.physical_dict = {}
        self.weather = carla.WeatherParameters()
        self.weather.sun_azimuth_angle = 45
        self.weather.sun_altitude_angle = 70

        self.scenario_declaration = {}
        self.struct_declaration = {}

        self._parse_osc2_configuration()

    def get_car_config(self, car_name: str) -> vehicles.Vehicle:
        return self.all_actors[car_name]

    def add_ego_vehicles(self, vc: vehicles.Vehicle):
        self.ego_vehicles.append(vc)
        self.all_actors[vc.get_name()] = vc

    def add_other_actors(self, npc: vehicles.Vehicle):
        self.other_actors.append(npc)
        self.all_actors[npc.get_name()] = npc

    def store_variable(self, vary):
        variable.Variable.set_arg(vary)

    class ConfigInit(ASTVisitor):
        def __init__(self, configInstance) -> None:
            super().__init__()
            self.father_ins = configInstance

        def visit_global_parameter_declaration(
            self, node: ast_node.GlobalParameterDeclaration
        ):
            para_name = node.field_name[0]
            para_type = ""
            para_value = ""
            arguments = self.visit_children(node)
            if isinstance(arguments, list) and len(arguments) == 2:
                para_type = arguments[0]
                para_value = arguments[1]
                self.father_ins.variables[para_name] = para_value
            elif isinstance(arguments, str):
                para_type = arguments
                if para_type in vehicle_type:
                    vehicle_class = getattr(vehicles, para_type)
                    v_ins = vehicle_class()

                    v_ins.set_name(para_name)
                    if para_name == "ego_vehicle":
                        self.father_ins.add_ego_vehicles(v_ins)
                    else:
                        self.father_ins.add_other_actors(v_ins)
                self.father_ins.variables[para_name] = para_type
            self.father_ins.store_variable(self.father_ins.variables)

        def visit_struct_declaration(self, node: ast_node.StructDeclaration):
            struct_name = node.struct_name
            self.father_ins.struct_declaration[struct_name] = node
            for child in node.get_children():
                if isinstance(child, ast_node.MethodDeclaration):
                    self.visit_method_declaration(child)

        def visit_scenario_declaration(self, node: ast_node.ScenarioDeclaration):
            scenario_name = node.qualified_behavior_name
            self.father_ins.scenario_declaration[scenario_name] = node
            if scenario_name != "top":
                return

            for child in node.get_children():
                if isinstance(child, ast_node.ParameterDeclaration):
                    self.visit_parameter_declaration(child)
                elif isinstance(child, ast_node.ModifierInvocation):
                    self.visit_modifier_invocation(child)
                elif isinstance(child, ast_node.VariableDeclaration):
                    self.visit_variable_declaration(child)
                elif isinstance(child, ast_node.EventDeclaration):
                    pass
                    # self.visit_event_declaration(child)
                elif isinstance(child, ast_node.DoDirective):
                    self.visit_do_directive(child)

        def visit_do_directive(self, node: ast_node.DoDirective):
            pass

        def visit_parameter_declaration(self, node: ast_node.ParameterDeclaration):
            para_name = node.field_name[0]
            para_type = ""
            para_value = ""
            arguments = self.visit_children(node)
            if isinstance(arguments, list) and len(arguments) == 2:
                para_type = arguments[0]
                para_value = arguments[1]
                if (
                    self.father_ins.variables.get(str(para_value)) is not None
                    and para_type not in vehicle_type
                ):
                    para_value = self.father_ins.variables.get(str(para_value))
                self.father_ins.variables[para_name] = para_value
            elif isinstance(arguments, str):
                para_type = arguments
                if para_type in vehicle_type:
                    vehicle_class = getattr(vehicles, para_type)
                    v_ins = vehicle_class()

                    # TODO: Analyzing and setting vehicle configuration parameters requires parsing the keep statement
                    v_ins.set_name(para_name)
                    if para_name == OSC2Helper.ego_name:
                        self.father_ins.add_ego_vehicles(v_ins)
                    else:
                        self.father_ins.add_other_actors(v_ins)
                self.father_ins.variables[para_name] = para_type
            self.father_ins.store_variable(self.father_ins.variables)

        def visit_variable_declaration(self, node: ast_node.VariableDeclaration):
            variable_name = node.field_name[0]
            # variable_type = ""
            variable_value = ""
            arguments = self.visit_children(node)
            if isinstance(arguments, list) and len(arguments) == 2:
                # variable_type = arguments[0]
                variable_value = arguments[1]
                if self.father_ins.variables.get(str(variable_value)) is not None:
                    variable_value = self.father_ins.variables.get(str(variable_value))
                self.father_ins.variables[variable_name] = variable_value
            elif isinstance(arguments, str):
                # variable_type = arguments
                self.father_ins.variables[variable_name] = variable_value
            self.father_ins.store_variable(self.father_ins.variables)

        def visit_modifier_invocation(self, node: ast_node.ModifierInvocation):
            function_name = node.modifier_name
            actor_name = node.actor
            arguments = self.visit_children(node)
            if hasattr(self.father_ins.path, function_name):
                path_function = getattr(self.father_ins.path, function_name)
                position_args = []
                keyword_args = {}
                if isinstance(arguments, List):
                    arguments = flat_list(arguments)
                    for arg in arguments:
                        if isinstance(arg, Tuple):
                            if self.father_ins.variables.get(arg[0]) is not None:
                                keyword_args[arg[0]] = self.father_ins.variables.get(
                                    arg[0]
                                )
                            else:
                                keyword_args[arg[0]] = arg[1]
                        else:
                            if self.father_ins.variables.get(arg) is not None:
                                position_args.append(self.father_ins.variables.get(arg))
                            else:
                                position_args.append(arg)
                else:
                    if self.father_ins.variables.get(arguments) is not None:
                        position_args.append(self.father_ins.variables.get(arguments))
                    else:
                        position_args.append(arguments)
                path_function(*position_args, **keyword_args)
            if actor_name == "ego_vehicle":
                if hasattr(vehicles.Vehicle, function_name):
                    position_args = []
                    keyword_args = {}
                    position_function = getattr(
                        self.father_ins.ego_vehicles[0], function_name
                    )

                    pos = misc.WorldPosition(0, 0, 0, 0, 0, 0)
                    position_cls = getattr(pos, "__init__")
                    if isinstance(arguments, List):
                        arguments = flat_list(arguments)
                        for arg in arguments:
                            if isinstance(arg, Tuple):
                                if isinstance(arg[1], Physical):
                                    keyword_args[arg[0]] = arg[1].gen_physical_value()
                            else:
                                if isinstance(arg, Physical):
                                    position_args.append(arg.gen_physical_value())
                    else:
                        if isinstance(arguments, Physical):
                            position_args.append(arguments.gen_physical_value())
                    position_cls(*position_args, **keyword_args)
                    position_function(pos)
                    self.father_ins.ego_vehicles[0].random_location = False

        def visit_event_declaration(self, node: ast_node.EventDeclaration):
            event_name = node.field_name
            arguments = self.visit_children(node)
            if hasattr(self.father_ins.event, event_name):
                # event_function = getattr(self.father_ins.event, event_name)
                position_args = []
                keyword_args = {}
                if isinstance(arguments, List):
                    arguments = flat_list(arguments)
                    for arg in arguments:
                        if isinstance(arg, Tuple):
                            if self.father_ins.variables.get(arg[0]) is not None:
                                keyword_args[arg[0]] = self.father_ins.variables.get(
                                    arg[0]
                                )
                            else:
                                keyword_args[arg[0]] = arg[1]
                        else:
                            if self.father_ins.variables.get(arg) is not None:
                                position_args.append(self.father_ins.variables.get(arg))
                            else:
                                position_args.append(arg)
                else:
                    if self.father_ins.variables.get(arguments) is not None:
                        position_args.append(self.father_ins.variables.get(arguments))
                    else:
                        position_args.append(arguments)

        def visit_method_declaration(self, node: ast_node.MethodDeclaration):
            pass

        def visit_method_body(self, node: ast_node.MethodBody):
            # type = node.type
            for child in node.get_children():
                if isinstance(child, ast_node.BinaryExpression):
                    self.visit_binary_expression(child)

        def visit_binary_expression(self, node: ast_node.BinaryExpression):
            arguments = [self.visit_children(node), node.operator]
            flat_arguments = flat_list(arguments)
            temp_stack = []
            for ex in flat_arguments:
                if ex in ('+', '-', '*', '/', '%'):
                    right = temp_stack.pop()
                    left = temp_stack.pop()
                    expression = left + " " + ex + " " + right
                    temp_stack.append(expression)
                else:
                    temp_stack.append(ex)
            flat_arguments.clear()
            binary_expression = temp_stack.pop()
            return binary_expression

        def visit_named_argument(self, node: ast_node.NamedArgument):
            return node.argument_name, self.visit_children(node)

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
                self.visit_children(node), self.father_ins.unit_dict[node.unit_name]
            )

        def visit_integer_literal(self, node: ast_node.IntegerLiteral):
            return int(node.value)

        def visit_float_literal(self, node: ast_node.FloatLiteral):
            return float(node.value)

        def visit_bool_literal(self, node: ast_node.BoolLiteral):
            return node.value

        def visit_string_literal(self, node: ast_node.StringLiteral):
            return node.value

        def visit_identifier(self, node: ast_node.Identifier):
            return node.name

        def visit_identifier_reference(self, node: ast_node.IdentifierReference):
            return node.name

        def visit_type(self, node: ast_node.Type):
            return node.type_name

        def visit_physical_type_declaration(
            self, node: ast_node.PhysicalTypeDeclaration
        ):
            si_base_exponent = {}
            arguments = self.visit_children(node)
            arguments = flat_list(arguments)
            if isinstance(arguments, Tuple):
                si_base_exponent[arguments[0]] = arguments[1]
            else:
                for elem in arguments:
                    si_base_exponent[elem[0]] = elem[1]
            self.father_ins.physical_dict[node.type_name] = PhysicalObject(
                node.type_name, si_base_exponent
            )

        def visit_unit_declaration(self, node: ast_node.UnitDeclaration):
            arguments = self.visit_children(node)
            arguments = flat_list(arguments)
            factor = 1.0
            offset = 0
            for elem in arguments:
                if elem[0] == "factor":
                    factor = elem[1]
                elif elem[0] == "offset":
                    offset = elem[1]
            self.father_ins.unit_dict[node.unit_name] = UnitObject(
                node.unit_name,
                self.father_ins.physical_dict[node.physical_name],
                factor,
                offset,
            )

        def visit_si_base_exponent(self, node: ast_node.SIBaseExponent):
            return node.unit_name, self.visit_children(node)

    def _parse_osc2_configuration(self):
        """
        Parse the given osc2 file, set and validate parameters
        """

        conf_visitor = self.ConfigInit(self)
        conf_visitor.visit(self.ast_tree)
        self._set_carla_town()

    def _set_carla_town(self):
        """ """
        self.town = self.path.get_map()

        # workaround for relative positions during init
        world = self.client.get_world()
        wmap = None
        if world:
            world.get_settings()
            wmap = world.get_map()
        if world is None or (
            wmap is not None and wmap.name.split("/")[-1] != self.town
        ):
            self.client.load_world(self.town)
            world = self.client.get_world()

            CarlaDataProvider.set_world(world)
            if CarlaDataProvider.is_sync_mode():
                world.tick()
            else:
                world.wait_for_tick()
        else:
            CarlaDataProvider.set_world(world)
