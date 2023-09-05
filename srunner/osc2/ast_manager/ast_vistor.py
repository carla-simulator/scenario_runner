import ast

import srunner.osc2.ast_manager.ast_node as ast_node
from srunner.osc2.ast_manager.ast_node import AST
from srunner.tools.osc2_helper import OSC2Helper


class BaseVisitor(object):
    def visit(self, tree):
        return tree.accept(self)

    def visit_children(self, node):
        result = self.default_result()
        n = node.get_child_count()
        for i in range(n):
            if not self.should_visit_next_child(node, result):
                return result

            c = node.get_child(i)
            if isinstance(c, AST):
                child_result = c.accept(self)
                result = self.aggregate_result(result, child_result)

        return result

    def default_result(self):
        return []

    def aggregate_result(self, aggregate, next_result):
        if aggregate:
            return [aggregate, next_result]
        else:
            return next_result

    def should_visit_next_child(self, node, current_result):
        return True


class ASTVisitor(BaseVisitor):
    def visit_compilation_unit(self, node: ast_node.CompilationUnit):
        return self.visit_children(node)

    def visit_physical_type_declaration(self, node: ast_node.PhysicalTypeDeclaration):
        return self.visit_children(node)

    def visit_unit_declaration(self, node: ast_node.UnitDeclaration):
        return self.visit_children(node)

    def visit_si_base_exponent(self, node: ast_node.SIBaseExponent):
        return self.visit_children(node)

    def visit_enum_declaration(self, node: ast_node.EnumDeclaration):
        return self.visit_children(node)

    def visit_enum_member_decl(self, node: ast_node.EnumMemberDecl):
        return self.visit_children(node)

    def visit_enum_value_reference(self, node: ast_node.EnumValueReference):
        return self.visit_children(node)

    def visit_inherts_condition(self, node: ast_node.InheritsCondition):
        return self.visit_children(node)

    def visit_struct_declaration(self, node: ast_node.StructDeclaration):
        return self.visit_children(node)

    def visit_struct_inherts(self, node: ast_node.StructInherts):
        return self.visit_children(node)

    def visit_actor_declaration(self, node: ast_node.ActorDeclaration):
        return self.visit_children(node)

    def visit_actor_inherts(self, node: ast_node.ActorInherts):
        return self.visit_children(node)

    def visit_scenario_declaration(self, node: ast_node.ScenarioDeclaration):
        return self.visit_children(node)

    def visit_scenario_inherts(self, node: ast_node.ScenarioInherts):
        return self.visit_children(node)

    def visit_action_declaration(self, node: ast_node.ActionDeclaration):
        return self.visit_children(node)

    def visit_action_inherts(self, node: ast_node.ActionInherts):
        return self.visit_children(node)

    def visit_modifier_declaration(self, node: ast_node.ModifierDeclaration):
        return self.visit_children(node)

    def visit_enum_type_extension(self, node: ast_node.EnumTypeExtension):
        return self.visit_children(node)

    def visit_structured_type_extension(self, node: ast_node.StructuredTypeExtension):
        return self.visit_children(node)

    def visit_global_parameter_declaration(
        self, node: ast_node.GlobalParameterDeclaration
    ):
        return self.visit_children(node)

    def visit_parameter_declaration(self, node: ast_node.ParameterDeclaration):
        return self.visit_children(node)

    def visit_parameter_reference(self, node: ast_node.ParameterReference):
        return self.visit_children(node)

    def visit_variable_declaration(self, node: ast_node.VariableDeclaration):
        return self.visit_children(node)

    def visit_event_declaration(self, node: ast_node.EventDeclaration):
        return self.visit_children(node)

    def visit_event_reference(self, node: ast_node.EventReference):
        return self.visit_children(node)

    def visit_event_field_declaration(self, node: ast_node.EventFieldDecl):
        return self.visit_children(node)

    def visit_event_condition(self, node: ast_node.EventCondition):
        return self.visit_children(node)

    def visit_method_declaration(self, node: ast_node.MethodDeclaration):
        return self.visit_children(node)

    def visit_method_body(self, node: ast_node.MethodBody):
        return self.visit_children(node)

    def visit_cover_declaration(self, node: ast_node.coverDeclaration):
        return self.visit_children(node)

    def visit_record_declaration(self, node: ast_node.recordDeclaration):
        return self.visit_children(node)

    def visit_argument(self, node: ast_node.Argument):
        return self.visit_children(node)

    def visit_named_argument(self, node: ast_node.NamedArgument):
        return self.visit_children(node)

    def visit_positional_argument(self, node: ast_node.PositionalArgument):
        return self.visit_children(node)

    def visit_variable_declaration(self, node: ast_node.VariableDeclaration):
        return self.visit_children(node)

    def visit_keep_constraint_declaration(
        self, node: ast_node.KeepConstraintDeclaration
    ):
        return self.visit_children(node)

    def visit_remove_default_declaration(self, node: ast_node.RemoveDefaultDeclaration):
        return self.visit_children(node)

    def visit_on_directive(self, node: ast_node.OnDirective):
        return self.visit_children(node)

    def visit_do_directive(self, node: ast_node.DoDirective):
        return self.visit_children(node)

    def visit_do_member(self, node: ast_node.DoMember):
        return self.visit_children(node)

    def visit_wait_directive(self, node: ast_node.WaitDirective):
        return self.visit_children(node)

    def visit_emit_directive(self, node: ast_node.EmitDirective):
        return self.visit_children(node)

    def visit_call_directive(self, node: ast_node.CallDirective):
        return self.visit_children(node)

    def visit_until_directive(self, node: ast_node.UntilDirective):
        return self.visit_children(node)

    def visit_behavior_invocation(self, node: ast_node.BehaviorInvocation):
        return self.visit_children(node)

    def visit_modifier_invocation(self, node: ast_node.ModifierInvocation):
        return self.visit_children(node)

    def visit_rise_expression(self, node: ast_node.RiseExpression):
        return self.visit_children(node)

    def visit_fall_expression(self, node: ast_node.FallExpression):
        return self.visit_children(node)

    def visit_elapsed_expression(self, node: ast_node.ElapsedExpression):
        return self.visit_children(node)

    def visit_every_expression(self, node: ast_node.EveryExpression):
        return self.visit_children(node)

    def visit_sample_expression(self, node: ast_node.SampleExpression):
        return self.visit_children(node)

    def visit_cast_expression(self, node: ast_node.CastExpression):
        return self.visit_children(node)

    def visit_type_test_expression(self, node: ast_node.TypeTestExpression):
        return self.visit_children(node)

    def visit_element_access_expression(self, node: ast_node.ElementAccessExpression):
        return self.visit_children(node)

    def visit_function_application_expression(
        self, node: ast_node.FunctionApplicationExpression
    ):
        return self.visit_children(node)

    def visit_binary_expression(self, node: ast_node.BinaryExpression):
        return self.visit_children(node)

    def visit_unary_expression(self, node: ast_node.UnaryExpression):
        return self.visit_children(node)

    def visit_ternary_expression(self, node: ast_node.TernaryExpression):
        return self.visit_children(node)

    def visit_list_expression(self, node: ast_node.ListExpression):
        return self.visit_children(node)

    def visit_range_expression(self, node: ast_node.RangeExpression):
        return self.visit_children(node)

    def visit_physical_literal(self, node: ast_node.PhysicalLiteral):
        return self.visit_children(node)

    def visit_integer_literal(self, node: ast_node.IntegerLiteral):
        return self.visit_children(node)

    def visit_float_literal(self, node: ast_node.FloatLiteral):
        return self.visit_children(node)

    def visit_bool_literal(self, node: ast_node.BoolLiteral):
        return self.visit_children(node)

    def visit_string_literal(self, node: ast_node.StringLiteral):
        return self.visit_children(node)

    def visit_type(self, node: ast_node.Type):
        return self.visit_children(node)

    def visit_identifier(self, node: ast_node.Identifier):
        return self.visit_children(node)

    def visit_identifier_reference(self, node: ast_node.IdentifierReference):
        return self.visit_children(node)
