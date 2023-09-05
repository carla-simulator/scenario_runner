import srunner.osc2.ast_manager.ast_node as ast_node
from srunner.osc2.ast_manager.ast_node import AST


class ASTListener:
    def enter_compilation_unit(self, node: ast_node.CompilationUnit):
        pass

    def exit_compilation_unit(self, node: ast_node.CompilationUnit):
        pass

    def enter_physical_type_declaration(self, node: ast_node.PhysicalTypeDeclaration):
        pass

    def exit_physical_type_declaration(self, node: ast_node.PhysicalTypeDeclaration):
        pass

    def enter_unit_declaration(self, node: ast_node.UnitDeclaration):
        pass

    def exit_unit_declaration(self, node: ast_node.UnitDeclaration):
        pass

    def enter_si_base_exponent(self, node: ast_node.SIBaseExponent):
        pass

    def exit_si_base_exponent(self, node: ast_node.SIBaseExponent):
        pass

    def enter_enum_declaration(self, node: ast_node.EnumDeclaration):
        pass

    def exit_enum_declaration(self, node: ast_node.EnumDeclaration):
        pass

    def enter_enum_member_decl(self, node: ast_node.EnumMemberDecl):
        pass

    def exit_enum_member_decl(self, node: ast_node.EnumMemberDecl):
        pass

    def enter_enum_value_reference(self, node: ast_node.EnumValueReference):
        pass

    def exit_enum_value_reference(self, node: ast_node.EnumValueReference):
        pass

    def enter_inherts_condition(self, node: ast_node.InheritsCondition):
        pass

    def exit_inherts_condition(self, node: ast_node.InheritsCondition):
        pass

    def enter_struct_declaration(self, node: ast_node.StructDeclaration):
        pass

    def exit_struct_declaration(self, node: ast_node.StructDeclaration):
        pass

    def enter_struct_inherts(self, node: ast_node.StructInherts):
        pass

    def exit_struct_inherts(self, node: ast_node.StructInherts):
        pass

    def enter_actor_declaration(self, node: ast_node.ActorDeclaration):
        pass

    def exit_actor_declaration(self, node: ast_node.ActorDeclaration):
        pass

    def enter_actor_inherts(self, node: ast_node.ActorInherts):
        pass

    def exit_actor_inherts(self, node: ast_node.ActorInherts):
        pass

    def enter_scenario_declaration(self, node: ast_node.ScenarioDeclaration):
        pass

    def exit_scenario_declaration(self, node: ast_node.ScenarioDeclaration):
        pass

    def enter_scenario_inherts(self, node: ast_node.ScenarioInherts):
        pass

    def exit_scenario_inherts(self, node: ast_node.ScenarioInherts):
        pass

    def enter_action_declaration(self, node: ast_node.ActionDeclaration):
        pass

    def exit_action_declaration(self, node: ast_node.ActionDeclaration):
        pass

    def enter_action_inherts(self, node: ast_node.ActionInherts):
        pass

    def exit_action_inherts(self, node: ast_node.ActionInherts):
        pass

    def enter_modifier_declaration(self, node: ast_node.ModifierDeclaration):
        pass

    def exit_modifier_declaration(self, node: ast_node.ModifierDeclaration):
        pass

    def enter_enum_type_extension(self, node: ast_node.EnumTypeExtension):
        pass

    def exit_enum_type_extension(self, node: ast_node.EnumTypeExtension):
        pass

    def enter_structured_type_extension(self, node: ast_node.StructuredTypeExtension):
        pass

    def exit_structured_type_extension(self, node: ast_node.StructuredTypeExtension):
        pass

    def enter_global_parameter_declaration(
        self, node: ast_node.GlobalParameterDeclaration
    ):
        pass

    def exit_global_parameter_declaration(
        self, node: ast_node.GlobalParameterDeclaration
    ):
        pass

    def enter_parameter_declaration(self, node: ast_node.GlobalParameterDeclaration):
        pass

    def exit_parameter_declaration(self, node: ast_node.GlobalParameterDeclaration):
        pass

    def enter_parameter_reference(self, node: ast_node.ParameterReference):
        pass

    def exit_parameter_reference(self, node: ast_node.ParameterReference):
        pass

    def enter_variable_declaration(self, node: ast_node.VariableDeclaration):
        pass

    def exit_variable_declaration(self, node: ast_node.VariableDeclaration):
        pass

    def enter_event_declaration(self, node: ast_node.EventDeclaration):
        pass

    def exit_event_declaration(self, node: ast_node.EventDeclaration):
        pass

    def enter_event_reference(self, node: ast_node.EventReference):
        pass

    def exit_event_reference(self, node: ast_node.EventReference):
        pass

    def enter_event_field_declaration(self, node: ast_node.EventFieldDecl):
        pass

    def exit_event_field_declaration(self, node: ast_node.EventFieldDecl):
        pass

    def enter_event_condition(self, node: ast_node.EventCondition):
        pass

    def exit_event_condition(self, node: ast_node.EventCondition):
        pass

    def enter_method_declaration(self, node: ast_node.MethodDeclaration):
        pass

    def exit_method_declaration(self, node: ast_node.MethodDeclaration):
        pass

    def enter_method_body(self, node: ast_node.MethodBody):
        pass

    def exit_method_body(self, node: ast_node.MethodBody):
        pass

    def enter_cover_declaration(self, node: ast_node.coverDeclaration):
        pass

    def exit_cover_declaration(self, node: ast_node.coverDeclaration):
        pass

    def enter_record_declaration(self, node: ast_node.recordDeclaration):
        pass

    def exit_record_declaration(self, node: ast_node.recordDeclaration):
        pass

    def enter_argument(self, node: ast_node.Argument):
        pass

    def exit_argument(self, node: ast_node.Argument):
        pass

    def enter_named_argument(self, node: ast_node.NamedArgument):
        pass

    def exit_named_argument(self, node: ast_node.NamedArgument):
        pass

    def enter_positional_argument(self, node: ast_node.PositionalArgument):
        pass

    def exit_positional_argument(self, node: ast_node.PositionalArgument):
        pass

    def enter_variable_declaration(self, node: ast_node.VariableDeclaration):
        pass

    def exit_variable_declaration(self, node: ast_node.VariableDeclaration):
        pass

    def enter_keep_constraint_declaration(
        self, node: ast_node.KeepConstraintDeclaration
    ):
        pass

    def exit_keep_constraint_declaration(
        self, node: ast_node.KeepConstraintDeclaration
    ):
        pass

    def enter_remove_default_declaration(self, node: ast_node.RemoveDefaultDeclaration):
        pass

    def exit_remove_default_declaration(self, node: ast_node.RemoveDefaultDeclaration):
        pass

    def enter_on_directive(self, node: ast_node.OnDirective):
        pass

    def exit_on_directive(self, node: ast_node.OnDirective):
        pass

    def enter_do_directive(self, node: ast_node.DoDirective):
        pass

    def exit_do_directive(self, node: ast_node.DoDirective):
        pass

    def enter_do_member(self, node: ast_node.DoMember):
        pass

    def exit_do_member(self, node: ast_node.DoMember):
        pass

    def enter_wait_directive(self, node: ast_node.WaitDirective):
        pass

    def exit_wait_directive(self, node: ast_node.WaitDirective):
        pass

    def enter_emit_directive(self, node: ast_node.EmitDirective):
        pass

    def exit_emit_directive(self, node: ast_node.EmitDirective):
        pass

    def enter_call_directive(self, node: ast_node.CallDirective):
        pass

    def exit_call_directive(self, node: ast_node.CallDirective):
        pass

    def enter_until_directive(self, node: ast_node.UntilDirective):
        pass

    def exit_until_directive(self, node: ast_node.UntilDirective):
        pass

    def enter_behavior_invocation(self, node: ast_node.BehaviorInvocation):
        pass

    def exit_behavior_invocation(self, node: ast_node.BehaviorInvocation):
        pass

    def enter_modifier_invocation(self, node: ast_node.ModifierInvocation):
        pass

    def exit_modifier_invocation(self, node: ast_node.ModifierInvocation):
        pass

    def enter_rise_expression(self, node: ast_node.RiseExpression):
        pass

    def exit_rise_expression(self, node: ast_node.RiseExpression):
        pass

    def enter_fall_expression(self, node: ast_node.FallExpression):
        pass

    def exit_fall_expression(self, node: ast_node.FallExpression):
        pass

    def enter_elapsed_expression(self, node: ast_node.ElapsedExpression):
        pass

    def exit_elapsed_expression(self, node: ast_node.ElapsedExpression):
        pass

    def enter_every_expression(self, node: ast_node.EveryExpression):
        pass

    def exit_every_expression(self, node: ast_node.EveryExpression):
        pass

    def enter_sample_expression(self, node: ast_node.SampleExpression):
        pass

    def exit_sample_expression(self, node: ast_node.SampleExpression):
        pass

    def enter_cast_expression(self, node: ast_node.CastExpression):
        pass

    def exit_cast_expression(self, node: ast_node.CastExpression):
        pass

    def enter_type_test_expression(self, node: ast_node.TypeTestExpression):
        pass

    def exit_type_test_expression(self, node: ast_node.TypeTestExpression):
        pass

    def enter_element_access_expression(self, node: ast_node.ElementAccessExpression):
        pass

    def exit_element_access_expression(self, node: ast_node.ElementAccessExpression):
        pass

    def enter_function_application_expression(
        self, node: ast_node.FunctionApplicationExpression
    ):
        pass

    def exit_function_application_expression(
        self, node: ast_node.FunctionApplicationExpression
    ):
        pass

    def enter_binary_expression(self, node: ast_node.BinaryExpression):
        pass

    def exit_binary_expression(self, node: ast_node.BinaryExpression):
        pass

    def enter_unary_expression(self, node: ast_node.UnaryExpression):
        pass

    def exit_unary_expression(self, node: ast_node.UnaryExpression):
        pass

    def enter_ternary_expression(self, node: ast_node.TernaryExpression):
        pass

    def exit_ternary_expression(self, node: ast_node.TernaryExpression):
        pass

    def enter_list_expression(self, node: ast_node.ListExpression):
        pass

    def exit_list_expression(self, node: ast_node.ListExpression):
        pass

    def enter_range_expression(self, node: ast_node.RangeExpression):
        pass

    def exit_range_expression(self, node: ast_node.RangeExpression):
        pass

    def enter_physical_literal(self, node: ast_node.PhysicalLiteral):
        pass

    def exit_physical_literal(self, node: ast_node.PhysicalLiteral):
        pass

    def enter_integer_literal(self, node: ast_node.IntegerLiteral):
        pass

    def exit_integer_literal(self, node: ast_node.IntegerLiteral):
        pass

    def enter_float_literal(self, node: ast_node.FloatLiteral):
        pass

    def exit_float_literal(self, node: ast_node.FloatLiteral):
        pass

    def enter_bool_literal(self, node: ast_node.BoolLiteral):
        pass

    def exit_bool_literal(self, node: ast_node.BoolLiteral):
        pass

    def enter_string_literal(self, node: ast_node.StringLiteral):
        pass

    def exit_string_literal(self, node: ast_node.StringLiteral):
        pass

    def enter_type(self, node: ast_node.Type):
        pass

    def exit_type(self, node: ast_node.Type):
        pass

    def enter_identifier(self, node: ast_node.Identifier):
        pass

    def exit_identifier(self, node: ast_node.Identifier):
        pass

    def enter_identifier_reference(self, node: ast_node.IdentifierReference):
        pass

    def exit_identifier_reference(self, node: ast_node.IdentifierReference):
        pass
