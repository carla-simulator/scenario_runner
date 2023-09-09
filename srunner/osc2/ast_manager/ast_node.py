from typing import List


class AST(object):
    def __init__(self):
        # line and column record the source location of the ast node
        self.__line = None
        self.__column = None
        self.__scope = None  # The scope of the record node
        self.__children = []  # child node

    def get_child_count(self):
        return len(self.__children)

    def get_children(self):
        if self.__children is not None:
            for child in self.__children:
                yield child

    def get_child(self, i):
        return self.__children[i]

    def set_children(self, *childs):
        for child in childs:
            if child is not None:
                if isinstance(child, List):
                    for ch in child:
                        self.__children.append(ch)
                else:
                    self.__children.append(child)

    def set_loc(self, line, column):
        self.__line = line
        self.__column = column

    def get_loc(self):
        return self.__line, self.__column

    def set_scope(self, scope):
        self.__scope = scope

    def get_scope(self):
        return self.__scope

    def accept(self, visitor):
        pass

    def enter_node(self, listener):
        pass

    def exit_node(self, listener):
        pass

    def __iter__(self):
        self.iter = iter(self.__children)
        return self.iter

    def __next__(self):
        return next(self.iter)

    def __str__(self) -> str:
        return self.__class__.__name__


# CompilationUnit
class CompilationUnit(AST):
    def __init__(self):
        super().__init__()

    def enter_node(self, listener):
        if hasattr(listener, "enter_compilation_unit"):
            listener.enter_compilation_unit(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_compilation_unit"):
            listener.exit_compilation_unit(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_compilation_unit"):
            return visitor.visit_compilation_unit(self)
        else:
            return visitor.visit_children(self)


# Declaration
class Declaration(AST):
    pass


# Declaration
class Expression(AST):
    pass


class PhysicalTypeDeclaration(Declaration):
    def __init__(self, type_name):
        super().__init__()
        self.type_name = type_name
        self.set_children(type_name)

    def enter_node(self, listener):
        if hasattr(listener, "enter_physical_type_declaration"):
            listener.enter_physical_type_declaration(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_physical_type_declaration"):
            listener.exit_physical_type_declaration(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_physical_type_declaration"):
            return visitor.visit_physical_type_declaration(self)
        else:
            return visitor.visit_children(self)


class UnitDeclaration(Declaration):
    def __init__(self, unit_name, physical_name):
        super().__init__()
        self.unit_name = unit_name
        self.physical_name = physical_name
        self.set_children(unit_name)

    def enter_node(self, listener):
        if hasattr(listener, "enter_unit_declaration"):
            listener.enter_unit_declaration(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_unit_declaration"):
            listener.exit_unit_declaration(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_unit_declaration"):
            return visitor.visit_unit_declaration(self)
        else:
            return visitor.visit_children(self)


class SIBaseExponent(AST):
    def __init__(self, unit_name):
        super().__init__()
        self.unit_name = unit_name
        self.set_children(unit_name)

    def enter_node(self, listener):
        if hasattr(listener, "enter_si_base_exponent"):
            listener.enter_si_base_exponent(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_si_base_exponent"):
            listener.exit_si_base_exponent(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_si_base_exponent"):
            return visitor.visit_si_base_exponent(self)
        else:
            return visitor.visit_children(self)


class EnumDeclaration(Declaration):
    def __init__(self, enum_name):
        super().__init__()
        self.enum_name = enum_name
        self.set_children(enum_name)

    def enter_node(self, listener):
        if hasattr(listener, "enter_enum_declaration"):
            listener.enter_enum_declaration(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_enum_declaration"):
            listener.exit_enum_declaration(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_enum_declaration"):
            return visitor.visit_enum_declaration(self)
        else:
            return visitor.visit_children(self)


class EnumMemberDecl(Declaration):
    def __init__(self, enum_member_name, num_member_value):
        super().__init__()
        self.enum_member_name = enum_member_name
        self.num_member_value = num_member_value
        self.set_children(enum_member_name, num_member_value)

    def enter_node(self, listener):
        if hasattr(listener, "enter_enum_member_decl"):
            listener.enter_enum_member_decl(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_enum_member_decl"):
            listener.exit_enum_member_decl(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_enum_member_decl"):
            return visitor.visit_enum_member_decl(self)
        else:
            return visitor.visit_children(self)


class EnumValueReference(AST):
    def __init__(self, enum_name, enum_member_name):
        super().__init__()
        self.enum_name = enum_name
        self.enum_member_name = enum_member_name
        self.set_children(enum_name, enum_member_name)

    def enter_node(self, listener):
        if hasattr(listener, "enter_enum_value_reference"):
            listener.enter_enum_value_reference(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_enum_value_reference"):
            listener.exit_enum_value_reference(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_enum_value_reference"):
            return visitor.visit_enum_value_reference(self)
        else:
            return visitor.visit_children(self)


class InheritsCondition(AST):
    def __init__(self, field_name, bool_literal):
        super().__init__()
        self.field_name = field_name
        self.set_children(field_name, bool_literal)

    def enter_node(self, listener):
        if hasattr(listener, "enter_inherits_condition"):
            listener.enter_inherits_condition(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_inherits_condition"):
            listener.exit_inherits_condition(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_inherits_condition"):
            return visitor.visit_inherits_condition(self)
        else:
            return visitor.visit_children(self)


class StructDeclaration(Declaration):
    def __init__(self, struct_name):
        super().__init__()
        self.struct_name = struct_name
        self.set_children(struct_name)

    def enter_node(self, listener):
        if hasattr(listener, "enter_struct_declaration"):
            listener.enter_struct_declaration(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_struct_declaration"):
            listener.exit_struct_declaration(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_struct_declaration"):
            return visitor.visit_struct_declaration(self)
        else:
            return visitor.visit_children(self)


class StructInherts(AST):
    def __init__(self, struct_name):
        super().__init__()
        self.struct_name = struct_name
        self.set_children(struct_name)

    def enter_node(self, listener):
        if hasattr(listener, "enter_struct_inherts"):
            listener.enter_struct_inherts(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_struct_inherts"):
            listener.exit_struct_inherts(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_struct_inherts"):
            return visitor.visit_struct_inherts(self)
        else:
            return visitor.visit_children(self)


class ActorDeclaration(Declaration):
    def __init__(self, actor_name):
        super().__init__()
        self.actor_name = actor_name
        self.set_children(actor_name)

    def enter_node(self, listener):
        if hasattr(listener, "enter_actor_declaration"):
            listener.enter_actor_declaration(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_actor_declaration"):
            listener.exit_actor_declaration(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_actor_declaration"):
            return visitor.visit_actor_declaration(self)
        else:
            return visitor.visit_children(self)


class ActorInherts(AST):
    def __init__(self, actor_name):
        super().__init__()
        self.actor_name = actor_name
        self.set_children(actor_name)

    def enter_node(self, listener):
        if hasattr(listener, "enter_actor_inherts"):
            listener.enter_actor_inherts(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_actor_inherts"):
            listener.exit_actor_inherts(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_actor_inherts"):
            return visitor.visit_actor_inherts(self)
        else:
            return visitor.visit_children(self)


class ScenarioDeclaration(Declaration):
    def __init__(self, qualified_behavior_name):
        super().__init__()
        self.qualified_behavior_name = qualified_behavior_name
        self.set_children(qualified_behavior_name)

    def enter_node(self, listener):
        if hasattr(listener, "enter_scenario_declaration"):
            listener.enter_scenario_declaration(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_scenario_declaration"):
            listener.exit_scenario_declaration(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_scenario_declaration"):
            return visitor.visit_scenario_declaration(self)
        else:
            return visitor.visit_children(self)


class ScenarioInherts(AST):
    def __init__(self, qualified_behavior_name):
        super().__init__()
        self.qualified_behavior_name = qualified_behavior_name
        self.set_children(qualified_behavior_name)

    def enter_node(self, listener):
        if hasattr(listener, "enter_scenario_inherts"):
            listener.enter_scenario_inherts(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_scenario_inherts"):
            listener.exit_scenario_inherts(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_scenario_inherts"):
            return visitor.visit_scenario_inherts(self)
        else:
            return visitor.visit_children(self)


class ActionDeclaration(Declaration):
    def __init__(self, qualified_behavior_name):
        super().__init__()
        self.qualified_behavior_name = qualified_behavior_name
        self.set_children(qualified_behavior_name)

    def enter_node(self, listener):
        if hasattr(listener, "enter_action_declaration"):
            listener.enter_action_declaration(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_action_declaration"):
            listener.exit_action_declaration(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_action_declaration"):
            return visitor.visit_action_declaration(self)
        else:
            return visitor.visit_children(self)


class ActionInherts(AST):
    def __init__(self, qualified_behavior_name):
        super().__init__()
        self.qualified_behavior_name = qualified_behavior_name
        self.set_children(qualified_behavior_name)

    def enter_node(self, listener):
        if hasattr(listener, "enter_action_inherts"):
            listener.enter_action_inherts(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_action_inherts"):
            listener.exit_action_inherts(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_action_inherts"):
            return visitor.visit_action_inherts(self)
        else:
            return visitor.visit_children(self)


class ModifierDeclaration(Declaration):
    def __init__(self, actor_name, modifier_name):
        super().__init__()
        self.actor_name = actor_name
        self.modifier_name = modifier_name
        if actor_name is not None:
            self.set_children(actor_name, modifier_name)
        else:
            self.set_children(modifier_name)

    def enter_node(self, listener):
        if hasattr(listener, "enter_modifier_declaration"):
            listener.enter_modifier_declaration(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_modifier_declaration"):
            listener.exit_modifier_declaration(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_modifier_declaration"):
            return visitor.visit_modifier_declaration(self)
        else:
            return visitor.visit_children(self)


class EnumTypeExtension(Declaration):
    def __init__(self, enum_name):
        super().__init__()
        self.enum_name = enum_name
        self.set_children(enum_name)

    def enter_node(self, listener):
        if hasattr(listener, "enter_enum_type_extension"):
            listener.enter_enum_type_extension(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_enum_type_extension"):
            listener.exit_enum_type_extension(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_enum_type_extension"):
            return visitor.visit_enum_type_extension(self)
        else:
            return visitor.visit_children(self)


class StructuredTypeExtension(Declaration):
    def __init__(self, type_name, qualified_behavior_name):
        super().__init__()
        self.type_name = type_name
        self.qualified_behavior_name = qualified_behavior_name
        if type_name is not None:
            self.set_children(type_name)
        else:
            self.set_children(qualified_behavior_name)

    def enter_node(self, listener):
        if hasattr(listener, "enter_structured_type_extension"):
            listener.enter_structured_type_extension(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_structured_type_extension"):
            listener.exit_structured_type_extension(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_structured_type_extension"):
            return visitor.visit_structured_type_extension(self)
        else:
            return visitor.visit_children(self)


class GlobalParameterDeclaration(Declaration):
    """
    children stores name, type, and default values, where default values are not required
    """

    def __init__(self, field_name, field_type):
        super().__init__()
        self.field_name = field_name
        self.field_type = field_type
        self.set_children(field_name)

    def enter_node(self, listener):
        if hasattr(listener, "enter_global_parameter_declaration"):
            listener.enter_global_parameter_declaration(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_global_parameter_declaration"):
            listener.exit_global_parameter_declaration(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_global_parameter_declaration"):
            return visitor.visit_global_parameter_declaration(self)
        else:
            return visitor.visit_children(self)


class ParameterDeclaration(Declaration):
    def __init__(self, field_name, field_type):
        super().__init__()
        self.field_name = field_name
        self.field_type = field_type
        self.set_children(field_name)

    def enter_node(self, listener):
        if hasattr(listener, "enter_parameter_declaration"):
            listener.enter_parameter_declaration(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_parameter_declaration"):
            listener.exit_parameter_declaration(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_parameter_declaration"):
            return visitor.visit_parameter_declaration(self)
        else:
            return visitor.visit_children(self)


class ParameterReference(AST):
    def __init__(self, field_name, field_access):
        super().__init__()
        self.field_name = field_name
        self.field_access = field_access
        self.set_children(field_name, field_access)

    def enter_node(self, listener):
        if hasattr(listener, "enter_parameter_reference"):
            listener.enter_parameter_reference(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_parameter_reference"):
            listener.exit_parameter_reference(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_parameter_reference"):
            return visitor.visit_parameter_reference(self)
        else:
            return visitor.visit_children(self)


class VariableDeclaration(Declaration):
    def __init__(self, field_name, field_type):
        super().__init__()
        self.field_name = field_name
        self.field_type = field_type
        self.set_children(field_name)

    def enter_node(self, listener):
        if hasattr(listener, "enter_variable_declaration"):
            listener.enter_variable_declaration(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_variable_declaration"):
            listener.exit_variable_declaration(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_variable_declaration"):
            return visitor.visit_variable_declaration(self)
        else:
            return visitor.visit_children(self)


class EventDeclaration(Declaration):
    def __init__(self, event_name):
        super().__init__()
        self.field_name = event_name
        self.set_children(event_name)

    def enter_node(self, listener):
        if hasattr(listener, "enter_event_declaration"):
            listener.enter_event_declaration(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_event_declaration"):
            listener.exit_event_declaration(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_event_declaration"):
            return visitor.visit_event_declaration(self)
        else:
            return visitor.visit_children(self)


class EventReference(AST):
    def __init__(self, event_path):
        super().__init__()
        self.event_path = event_path
        self.set_children(event_path)

    def enter_node(self, listener):
        if hasattr(listener, "enter_event_reference"):
            listener.enter_event_reference(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_event_reference"):
            listener.exit_event_reference(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_event_reference"):
            return visitor.visit_event_reference(self)
        else:
            return visitor.visit_children(self)


class EventFieldDecl(AST):
    def __init__(self, event_field_name):
        super().__init__()
        self.event_field_name = event_field_name
        self.set_children(event_field_name)

    def enter_node(self, listener):
        if hasattr(listener, "enter_event_field_declaration"):
            listener.enter_event_field_declaration(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_event_field_declaration"):
            listener.exit_event_field_declaration(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_event_field_declaration"):
            return visitor.visit_event_field_declaration(self)
        else:
            return visitor.visit_children(self)


class EventCondition(AST):
    def __init__(self):
        super().__init__()

    def enter_node(self, listener):
        if hasattr(listener, "enter_event_condition"):
            listener.enter_event_condition(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_event_condition"):
            listener.exit_event_condition(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_event_condition"):
            return visitor.visit_event_condition(self)
        else:
            return visitor.visit_children(self)


class MethodDeclaration(Declaration):
    def __init__(self, method_name, return_type):
        super().__init__()
        self.method_name = method_name
        self.return_type = return_type
        self.set_children(method_name)

    def enter_node(self, listener):
        if hasattr(listener, "enter_method_declaration"):
            listener.enter_method_declaration(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_method_declaration"):
            listener.exit_method_declaration(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_method_declaration"):
            return visitor.visit_method_declaration(self)
        else:
            return visitor.visit_children(self)


class MethodBody(AST):
    """
    There are three types of method, expression, undefined, external
    In the children of this node, the type and the concrete method body are stored, in turn
    """

    def __init__(self, qualifier, type, external_name):
        super().__init__()
        self.qualifier = qualifier
        self.type = type
        self.external_name = external_name
        self.set_children(qualifier, external_name)

    def enter_node(self, listener):
        if hasattr(listener, "enter_method_body"):
            listener.enter_method_body(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_method_body"):
            listener.exit_method_body(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_method_body"):
            return visitor.visit_method_body(self)
        else:
            return visitor.visit_children(self)


class coverDeclaration(Declaration):
    """
    When override a 'cover', target name is none,
    but must have an argument with name 'override'.
    """

    def __init__(self, target_name):
        super().__init__()
        self.target_name = target_name
        self.set_children(target_name)

    def enter_node(self, listener):
        if hasattr(listener, "enter_cover_declaration"):
            listener.enter_cover_declaration(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_cover_declaration"):
            listener.exit_cover_declaration(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_cover_declaration"):
            return visitor.visit_cover_declaration(self)
        else:
            return visitor.visit_children(self)


class recordDeclaration(Declaration):
    """
    When override a 'record', target name is none,
    but must have an argument with name 'override'.
    """

    def __init__(self, target_name):
        super().__init__()
        self.target_name = target_name
        self.set_children(target_name)

    def enter_node(self, listener):
        if hasattr(listener, "enter_record_declaration"):
            listener.enter_record_declaration(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_record_declaration"):
            listener.exit_record_declaration(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_record_declaration"):
            return visitor.visit_record_declaration(self)
        else:
            return visitor.visit_children(self)


class Argument(AST):
    def __init__(self, argument_name, argument_type, default_value):
        super().__init__()
        self.argument_name = argument_name
        self.argument_type = argument_type
        self.default_value = default_value
        if default_value is not None:
            self.set_children(argument_name)
        else:
            self.set_children(argument_name)

    def enter_node(self, listener):
        if hasattr(listener, "enter_argument"):
            listener.enter_argument(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_argument"):
            listener.exit_argument(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_argument"):
            return visitor.visit_argument(self)
        else:
            return visitor.visit_children(self)


class NamedArgument(AST):
    def __init__(self, argument_name):
        super().__init__()
        self.argument_name = argument_name
        self.set_children(argument_name)

    def enter_node(self, listener):
        if hasattr(listener, "enter_named_argument"):
            listener.enter_named_argument(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_named_argument"):
            listener.exit_named_argument(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_named_argument"):
            return visitor.visit_named_argument(self)
        else:
            return visitor.visit_children(self)


class PositionalArgument(AST):
    def __init__(self):
        super().__init__()

    def enter_node(self, listener):
        if hasattr(listener, "enter_positional_argument"):
            listener.enter_positional_argument(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_positional_argument"):
            listener.exit_positional_argument(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_positional_argument"):
            return visitor.visit_positional_argument(self)
        else:
            return visitor.visit_children(self)


class VariableDeclaration(Declaration):
    """
    'var' fieldName (',' fieldName)* ':' typeDeclarator ('=' (sampleExpression | valueExp) )? NEWLINE;
    """

    def __init__(self, field_name, field_type):
        super().__init__()
        self.field_name = field_name
        self.field_type = field_type
        self.set_children(field_name)

    def enter_node(self, listener):
        if hasattr(listener, "enter_variable_declaration"):
            listener.enter_variable_declaration(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_variable_declaration"):
            listener.exit_variable_declaration(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_variable_declaration"):
            return visitor.visit_variable_declaration(self)
        else:
            return visitor.visit_children(self)


class KeepConstraintDeclaration(Declaration):
    def __init__(self, constraint_qualifier):
        super().__init__()
        self.constraint_qualifier = constraint_qualifier
        self.set_children(constraint_qualifier)

    def enter_node(self, listener):
        if hasattr(listener, "enter_keep_constraint_declaration"):
            listener.enter_keep_constraint_declaration(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_keep_constraint_declaration"):
            listener.exit_keep_constraint_declaration(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_keep_constraint_declaration"):
            return visitor.visit_keep_constraint_declaration(self)
        else:
            return visitor.visit_children(self)


class RemoveDefaultDeclaration(Declaration):
    def __init__(self):
        super().__init__()

    def enter_node(self, listener):
        if hasattr(listener, "enter_remove_default_declaration"):
            listener.enter_remove_default_declaration(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_remove_default_declaration"):
            listener.exit_remove_default_declaration(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_remove_default_declaration"):
            return visitor.visit_remove_default_declaration(self)
        else:
            return visitor.visit_children(self)


class OnDirective(AST):
    def __init__(self):
        super().__init__()

    def enter_node(self, listener):
        if hasattr(listener, "enter_on_directive"):
            listener.enter_on_directive(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_on_directive"):
            listener.exit_on_directive(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_on_directive"):
            return visitor.visit_on_directive(self)
        else:
            return visitor.visit_children(self)


class DoDirective(AST):
    def __init__(self):
        super().__init__()

    def enter_node(self, listener):
        if hasattr(listener, "enter_do_directive"):
            listener.enter_do_directive(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_do_directive"):
            listener.exit_do_directive(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_do_directive"):
            return visitor.visit_do_directive(self)
        else:
            return visitor.visit_children(self)


class DoMember(AST):
    def __init__(self, label_name, composition_operator):
        super().__init__()
        self.label_name = label_name
        self.composition_operator = composition_operator
        self.set_children(label_name, composition_operator)

    def enter_node(self, listener):
        if hasattr(listener, "enter_do_member"):
            listener.enter_do_member(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_do_member"):
            listener.exit_do_member(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_do_member"):
            return visitor.visit_do_member(self)
        else:
            return visitor.visit_children(self)


class WaitDirective(AST):
    def __init__(self):
        super().__init__()

    def enter_node(self, listener):
        if hasattr(listener, "enter_wait_directive"):
            listener.enter_wait_directive(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_wait_directive"):
            listener.exit_wait_directive(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_wait_directive"):
            return visitor.visit_wait_directive(self)
        else:
            return visitor.visit_children(self)


class EmitDirective(AST):
    def __init__(self, event_name):
        super().__init__()
        self.event_name = event_name
        self.set_children(event_name)

    def enter_node(self, listener):
        if hasattr(listener, "enter_emit_directive"):
            listener.enter_emit_directive(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_emit_directive"):
            listener.exit_emit_directive(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_emit_directive"):
            return visitor.visit_emit_directive(self)
        else:
            return visitor.visit_children(self)


class CallDirective(AST):
    def __init__(self, method_name):
        super().__init__()
        self.method_name = method_name

    def enter_node(self, listener):
        if hasattr(listener, "enter_call_directive"):
            listener.enter_call_directive(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_call_directive"):
            listener.exit_call_directive(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_call_directive"):
            return visitor.visit_call_directive(self)
        else:
            return visitor.visit_children(self)


class UntilDirective(AST):
    def __init__(self):
        super().__init__()

    def enter_node(self, listener):
        if hasattr(listener, "enter_until_directive"):
            listener.enter_until_directive(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_until_directive"):
            listener.exit_until_directive(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_until_directive"):
            return visitor.visit_until_directive(self)
        else:
            return visitor.visit_children(self)


class BehaviorInvocation(AST):
    def __init__(self, actor, behavior_name):
        super().__init__()
        self.actor = actor
        self.behavior_name = behavior_name
        self.set_children(actor, behavior_name)

    def enter_node(self, listener):
        if hasattr(listener, "enter_behavior_invocation"):
            listener.enter_behavior_invocation(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_behavior_invocation"):
            listener.exit_behavior_invocation(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_behavior_invocation"):
            return visitor.visit_behavior_invocation(self)
        else:
            return visitor.visit_children(self)


class ModifierInvocation(AST):
    def __init__(self, actor, modifier_name):
        super().__init__()
        self.actor = actor
        self.modifier_name = modifier_name
        self.set_children(actor, modifier_name)

    def enter_node(self, listener):
        if hasattr(listener, "enter_modifier_invocation"):
            listener.enter_modifier_invocation(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_modifier_invocation"):
            listener.exit_modifier_invocation(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_modifier_invocation"):
            return visitor.visit_modifier_invocation(self)
        else:
            return visitor.visit_children(self)


class RiseExpression(Expression):
    def __init__(self):
        super().__init__()

    def enter_node(self, listener):
        if hasattr(listener, "enter_rise_expression"):
            listener.enter_rise_expression(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_rise_expression"):
            listener.exit_rise_expression(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_rise_expression"):
            return visitor.visit_rise_expression(self)
        else:
            return visitor.visit_children(self)


class FallExpression(Expression):
    def __init__(self):
        super().__init__()

    def enter_node(self, listener):
        if hasattr(listener, "enter_fall_expression"):
            listener.enter_rise_expression(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_fall_expression"):
            listener.exit_fall_expression(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_fall_expression"):
            return visitor.visit_fall_expression(self)
        else:
            return visitor.visit_children(self)


class ElapsedExpression(Expression):
    def __init__(self):
        super().__init__()

    def enter_node(self, listener):
        if hasattr(listener, "enter_elapsed_expression"):
            listener.enter_rise_expression(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_elapsed_expression"):
            listener.exit_fall_expression(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_elapsed_expression"):
            return visitor.visit_fall_expression(self)
        else:
            return visitor.visit_children(self)


class EveryExpression(Expression):
    """
    In every expression, there are at most two children among its children,
    The first child is the primary expression and the second child is the offset expression
    'every' OPEN_PAREN durationExpression (',' 'offset' ':' durationExpression)? CLOSE_PAREN;
    """

    def __init__(self):
        super().__init__()

    def enter_node(self, listener):
        if hasattr(listener, "enter_every_expression"):
            listener.enter_every_expression(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_every_expression"):
            listener.exit_every_expression(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_every_expression"):
            return visitor.visit_every_expression(self)
        else:
            return visitor.visit_children(self)


class SampleExpression(Expression):
    """
    In the sample expression, there are at most three children among its children，
    'sample' OPEN_PAREN expression ',' eventSpecification (',' defaultValue)? CLOSE_PAREN;
    """

    def __init__(self):
        super().__init__()

    def enter_node(self, listener):
        if hasattr(listener, "enter_sample_expression"):
            listener.enter_sample_expression(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_sample_expression"):
            listener.exit_sample_expression(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_sample_expression"):
            return visitor.visit_sample_expression(self)
        else:
            return visitor.visit_children(self)


class CastExpression(Expression):
    """ """

    def __init__(self, object, target_type):
        super().__init__()
        self.object = object
        self.target_type = target_type

    def enter_node(self, listener):
        if hasattr(listener, "enter_cast_expression"):
            listener.enter_cast_expression(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_cast_expression"):
            listener.exit_cast_expression(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_cast_expression"):
            return visitor.visit_cast_expression(self)
        else:
            return visitor.visit_children(self)


class TypeTestExpression(Expression):
    """ """

    def __init__(self, object, target_type):
        super().__init__()
        self.object = object
        self.target_type = target_type

    def enter_node(self, listener):
        if hasattr(listener, "enter_type_test_expression"):
            listener.enter_type_test_expression(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_type_test_expression"):
            listener.exit_type_test_expression(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_type_test_expression"):
            return visitor.visit_type_test_expression(self)
        else:
            return visitor.visit_children(self)


class ElementAccessExpression(Expression):
    """ """

    def __init__(self, list_name, index):
        super().__init__()
        self.list_name = list_name
        self.index = index

    def enter_node(self, listener):
        if hasattr(listener, "enter_element_access_expression"):
            listener.enter_element_access_expression(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_element_access_expression"):
            listener.exit_element_access_expression(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_element_access_expression"):
            return visitor.visit_element_access_expression(self)
        else:
            return visitor.visit_children(self)


class FunctionApplicationExpression(Expression):
    """
    In a functionApplication expression, store the method name and its arguments in the children.
    Method names are represented by identifier nodes
    """

    def __init__(self, func_name):
        super().__init__()
        self.func_name = func_name

    def enter_node(self, listener):
        if hasattr(listener, "enter_function_application_expression"):
            listener.enter_function_application_expression(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_function_application_expression"):
            listener.exit_function_application_expression(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_function_application_expression"):
            return visitor.visit_function_application_expression(self)
        else:
            return visitor.visit_children(self)


class FieldAccessExpression(Expression):
    def __init__(self, field_name):
        super().__init__()
        self.field_name = field_name
        self.set_children(field_name)

    def enter_node(self, listener):
        if hasattr(listener, "enter_field_access_expression"):
            listener.enter_field_access_expression(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_field_access_expression"):
            listener.exit_field_access_expression(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_field_access_expression"):
            return visitor.visit_field_access_expression(self)
        else:
            return visitor.visit_children(self)


class BinaryExpression(Expression):
    """
    In the children of this node, the operator, left expression, and right expression are stored, in order
    """

    def __init__(self, operator):
        super().__init__()
        self.operator = operator
        self.set_children(operator)

    def enter_node(self, listener):
        if hasattr(listener, "enter_binary_expression"):
            listener.enter_binary_expression(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_binary_expression"):
            listener.exit_binary_expression(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_binary_expression"):
            return visitor.visit_binary_expression(self)
        else:
            return visitor.visit_children(self)


class UnaryExpression(Expression):
    """
    In the children of this node, operators are stored, followed by expressions
    """

    def __init__(self, operator):
        super().__init__()
        self.operator = operator
        self.set_children(operator)

    def enter_node(self, listener):
        if hasattr(listener, "enter_unary_expression"):
            listener.enter_unary_expression(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_unary_expression"):
            listener.exit_unary_expression(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_unary_expression"):
            return visitor.visit_unary_expression(self)
        else:
            return visitor.visit_children(self)


class TernaryExpression(Expression):
    """
    In the children of this node, the conditional expression is stored,
    followed by the left expression, and then the right expression
    """

    def __init__(self):
        super().__init__()

    def enter_node(self, listener):
        if hasattr(listener, "enter_ternary_expression"):
            listener.enter_ternary_expression(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_ternary_expression"):
            listener.exit_ternary_expression(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_ternary_expression"):
            return visitor.visit_ternary_expression(self)
        else:
            return visitor.visit_children(self)


class LogicalExpression(Expression):
    """
    In logical expressions, for the same operator, such as a=>b=>c, the operands of the expression are placed in children.
    It is not divided into multiple binary expressions like binary expressions
    """

    def __init__(self, operator):
        super().__init__()
        self.operator = operator
        self.set_children(operator)

    def enter_node(self, listener):
        if hasattr(listener, "enter_logical_expression"):
            listener.enter_logical_expression(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_logical_expression"):
            listener.exit_logical_expression(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_logical_expression"):
            return visitor.visit_logical_expression(self)
        else:
            return visitor.visit_children(self)


class RelationExpression(Expression):
    def __init__(self, operator):
        super().__init__()
        self.operator = operator
        self.set_children(operator)

    def enter_node(self, listener):
        if hasattr(listener, "enter_relation_expression"):
            listener.enter_relation_expression(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_relation_expression"):
            listener.exit_relation_expression(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_relation_expression"):
            return visitor.visit_relation_expression(self)
        else:
            return visitor.visit_children(self)


class ListExpression(Expression):
    """
    In a list expression, each child node must be of the same type
    """

    def __init__(self):
        super().__init__()

    def enter_node(self, listener):
        if hasattr(listener, "enter_list_expression"):
            listener.enter_list_expression(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_list_expression"):
            listener.exit_list_literal(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_list_expression"):
            return visitor.visit_list_expression(self)
        else:
            return visitor.visit_children(self)


class RangeExpression(Expression):
    """
    In a range expression, the first and second expressions are stored in the child nodes
    """

    def __init__(self):
        super().__init__()

    def enter_node(self, listener):
        if hasattr(listener, "enter_range_expression"):
            listener.enter_range_expression(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_range_expression"):
            listener.exit_range_expression(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_range_expression"):
            return visitor.visit_range_expression(self)
        else:
            return visitor.visit_children(self)


class PhysicalLiteral(AST):
    def __init__(self, unit_name, value):
        super().__init__()
        self.value = value
        self.unit_name = unit_name
        self.set_children(unit_name)

    def enter_node(self, listener):
        if hasattr(listener, "enter_physical_literal"):
            listener.enter_physical_literal(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_physical_literal"):
            listener.exit_physical_literal(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_physical_literal"):
            return visitor.visit_physical_literal(self)
        else:
            return visitor.visit_children(self)


class IntegerLiteral(AST):
    def __init__(self, type, value):
        super().__init__()
        self.type = type  # uint, hex, int
        self.value = value
        self.set_children(type, value)

    def enter_node(self, listener):
        if hasattr(listener, "enter_integer_literal"):
            listener.enter_integer_literal(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_integer_literal"):
            listener.exit_integer_literal(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_integer_literal"):
            return visitor.visit_integer_literal(self)
        else:
            return visitor.visit_children(self)


class FloatLiteral(AST):
    def __init__(self, value):
        super().__init__()
        self.value = value
        self.set_children(value)

    def enter_node(self, listener):
        if hasattr(listener, "enter_float_literal"):
            listener.enter_float_literal(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_float_literal"):
            listener.exit_float_literal(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_float_literal"):
            return visitor.visit_float_literal(self)
        else:
            return visitor.visit_children(self)


class BoolLiteral(AST):
    def __init__(self, value):
        super().__init__()
        self.value = value
        self.set_children(value)

    def enter_node(self, listener):
        if hasattr(listener, "enter_bool_literal"):
            listener.enter_bool_literal(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_bool_literal"):
            listener.exit_bool_literal(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_bool_literal"):
            return visitor.visit_bool_literal(self)
        else:
            return visitor.visit_children(self)


class StringLiteral(AST):
    def __init__(self, value):
        super().__init__()
        self.value = value
        self.set_children(value)

    def enter_node(self, listener):
        if hasattr(listener, "enter_string_literal"):
            listener.enter_string_literal(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_string_literal"):
            listener.exit_string_literal(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_string_literal"):
            return visitor.visit_string_literal(self)
        else:
            return visitor.visit_children(self)


class Type(AST):
    def __init__(self, type_name):
        super().__init__()
        self.type_name = type_name
        self.set_children(type_name)

    def enter_node(self, listener):
        if hasattr(listener, "enter_type"):
            listener.enter_type(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_type"):
            listener.exit_type(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_type"):
            return visitor.visit_type(self)
        else:
            return visitor.visit_type(self)


class Identifier(AST):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.set_children(name)

    def enter_node(self, listener):
        if hasattr(listener, "enter_identifier"):
            listener.enter_identifier(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_identifier"):
            listener.exit_identifier(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_identifier"):
            return visitor.visit_identifier(self)
        else:
            return visitor.visit_children(self)


class IdentifierReference(AST):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.set_children(name)

    def enter_node(self, listener):
        if hasattr(listener, "enter_identifier_reference"):
            listener.enter_identifier_reference(self)

    def exit_node(self, listener):
        if hasattr(listener, "exit_identifier_reference"):
            listener.exit_identifier_reference(self)

    def accept(self, visitor):
        if hasattr(visitor, "visit_identifier_reference"):
            return visitor.visit_identifier_reference(self)
        else:
            return visitor.visit_identifier_reference(self)
