import copy
from pydoc import resolve

import srunner.osc2.ast_manager.ast_node as ast_node
from srunner.osc2.osc2_parser.OpenSCENARIO2Listener import OpenSCENARIO2Listener
from srunner.osc2.osc2_parser.OpenSCENARIO2Parser import OpenSCENARIO2Parser
from srunner.osc2.symbol_manager.action_symbol import ActionSymbol
from srunner.osc2.symbol_manager.actor_symbol import ActorSymbol
from srunner.osc2.symbol_manager.argument_symbol import *
from srunner.osc2.symbol_manager.constraint_decl_scope import *
from srunner.osc2.symbol_manager.do_directive_scope import *
from srunner.osc2.symbol_manager.doMember_symbol import DoMemberSymbol
from srunner.osc2.symbol_manager.enum_symbol import *
from srunner.osc2.symbol_manager.event_symbol import *
from srunner.osc2.symbol_manager.global_scope import GlobalScope
from srunner.osc2.symbol_manager.inherits_condition_symbol import *
from srunner.osc2.symbol_manager.method_symbol import MethodSymbol
from srunner.osc2.symbol_manager.modifier_symbol import *
from srunner.osc2.symbol_manager.parameter_symbol import ParameterSymbol
from srunner.osc2.symbol_manager.physical_type_symbol import PhysicalTypeSymbol
from srunner.osc2.symbol_manager.qualifiedBehavior_symbol import QualifiedBehaviorSymbol
from srunner.osc2.symbol_manager.scenario_symbol import ScenarioSymbol
from srunner.osc2.symbol_manager.si_exponent_symbol import (
    SiBaseExponentListScope,
    SiExpSymbol,
)
from srunner.osc2.symbol_manager.struct_symbol import StructSymbol
from srunner.osc2.symbol_manager.typed_symbol import *
from srunner.osc2.symbol_manager.unit_symbol import UnitSymbol
from srunner.osc2.symbol_manager.variable_symbol import VariableSymbol
from srunner.osc2.symbol_manager.wait_symbol import *
from srunner.osc2.utils.log_manager import *
from srunner.osc2.utils.tools import *


class ASTBuilder(OpenSCENARIO2Listener):
    def __init__(self):
        self.__global_scope = None  # Global scope
        self.__current_scope = None  # Current scope
        self.__node_stack = []
        self.__cur_node = None
        self.ast = None

    def get_ast(self):
        return self.ast

    def get_symbol(self):
        return self.__current_scope

    # Enter a parse tree produced by OpenSCENARIO2Parser#osc_file.
    def enterOsc_file(self, ctx: OpenSCENARIO2Parser.Osc_fileContext):
        self.__global_scope = GlobalScope(None)
        self.__current_scope = self.__global_scope

        node = ast_node.CompilationUnit()
        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(self.__current_scope)
        self.__node_stack.append(node)
        self.__cur_node = node
        self.ast = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#osc_file.
    def exitOsc_file(self, ctx: OpenSCENARIO2Parser.Osc_fileContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#preludeStatement.
    def enterPreludeStatement(self, ctx: OpenSCENARIO2Parser.PreludeStatementContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#preludeStatement.
    def exitPreludeStatement(self, ctx: OpenSCENARIO2Parser.PreludeStatementContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#importStatement.
    def enterImportStatement(self, ctx: OpenSCENARIO2Parser.ImportStatementContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#importStatement.
    def exitImportStatement(self, ctx: OpenSCENARIO2Parser.ImportStatementContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#importReference.
    def enterImportReference(self, ctx: OpenSCENARIO2Parser.ImportReferenceContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#importReference.
    def exitImportReference(self, ctx: OpenSCENARIO2Parser.ImportReferenceContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#structuredIdentifier.
    def enterStructuredIdentifier(
        self, ctx: OpenSCENARIO2Parser.StructuredIdentifierContext
    ):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#structuredIdentifier.
    def exitStructuredIdentifier(
        self, ctx: OpenSCENARIO2Parser.StructuredIdentifierContext
    ):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#oscDeclaration.
    def enterOscDeclaration(self, ctx: OpenSCENARIO2Parser.OscDeclarationContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#oscDeclaration.
    def exitOscDeclaration(self, ctx: OpenSCENARIO2Parser.OscDeclarationContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#physicalTypeDeclaration.
    def enterPhysicalTypeDeclaration(
        self, ctx: OpenSCENARIO2Parser.PhysicalTypeDeclarationContext
    ):
        self.__node_stack.append(self.__cur_node)
        type_name = ctx.physicalTypeName().getText()

        physical_type = PhysicalTypeSymbol(type_name, self.__current_scope)
        self.__current_scope.define(physical_type, ctx.start)
        self.__current_scope = physical_type

        node = ast_node.PhysicalTypeDeclaration(type_name)
        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(self.__current_scope)

        self.__cur_node.set_children(node)
        self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#physicalTypeDeclaration.
    def exitPhysicalTypeDeclaration(
        self, ctx: OpenSCENARIO2Parser.PhysicalTypeDeclarationContext
    ):
        self.__cur_node = self.__node_stack.pop()
        self.__current_scope = self.__current_scope.get_enclosing_scope()

    # Enter a parse tree produced by OpenSCENARIO2Parser#physicalTypeName.
    def enterPhysicalTypeName(self, ctx: OpenSCENARIO2Parser.PhysicalTypeNameContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#physicalTypeName.
    def exitPhysicalTypeName(self, ctx: OpenSCENARIO2Parser.PhysicalTypeNameContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#baseUnitSpecifier.
    def enterBaseUnitSpecifier(self, ctx: OpenSCENARIO2Parser.BaseUnitSpecifierContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#baseUnitSpecifier.
    def exitBaseUnitSpecifier(self, ctx: OpenSCENARIO2Parser.BaseUnitSpecifierContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#sIBaseUnitSpecifier.
    def enterSIBaseUnitSpecifier(
        self, ctx: OpenSCENARIO2Parser.SIBaseUnitSpecifierContext
    ):
        si_scope = SiBaseExponentListScope(self.__current_scope)
        self.__current_scope.define(si_scope, ctx.start)
        self.__current_scope = si_scope

    # Exit a parse tree produced by OpenSCENARIO2Parser#sIBaseUnitSpecifier.
    def exitSIBaseUnitSpecifier(
        self, ctx: OpenSCENARIO2Parser.SIBaseUnitSpecifierContext
    ):
        self.__current_scope = self.__current_scope.get_enclosing_scope()

    # Enter a parse tree produced by OpenSCENARIO2Parser#unitDeclaration.
    def enterUnitDeclaration(self, ctx: OpenSCENARIO2Parser.UnitDeclarationContext):
        self.__node_stack.append(self.__cur_node)
        unit_name = ctx.Identifier().getText()

        physical_name = ctx.physicalTypeName().getText()
        # Find the physical quantity and define the unit of physical quantity
        if self.__current_scope.resolve(physical_name):
            pass
        else:
            msg = "PhysicalType: " + physical_name + " is not defined!"
            LOG_ERROR(msg, ctx.start)

        unit = UnitSymbol(unit_name, self.__current_scope, physical_name)
        self.__current_scope.define(unit, ctx.start)
        self.__current_scope = unit

        node = ast_node.UnitDeclaration(unit_name, physical_name)
        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(self.__current_scope)

        self.__cur_node.set_children(node)
        self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#unitDeclaration.
    def exitUnitDeclaration(self, ctx: OpenSCENARIO2Parser.UnitDeclarationContext):
        self.__cur_node = self.__node_stack.pop()
        self.__current_scope = self.__current_scope.get_enclosing_scope()

    # Enter a parse tree produced by OpenSCENARIO2Parser#unitSpecifier.
    def enterUnitSpecifier(self, ctx: OpenSCENARIO2Parser.UnitSpecifierContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#unitSpecifier.
    def exitUnitSpecifier(self, ctx: OpenSCENARIO2Parser.UnitSpecifierContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#sIUnitSpecifier.
    def enterSIUnitSpecifier(self, ctx: OpenSCENARIO2Parser.SIUnitSpecifierContext):
        si_scope = SiBaseExponentListScope(self.__current_scope)
        self.__current_scope.define(si_scope, ctx.start)
        self.__current_scope = si_scope

    # Exit a parse tree produced by OpenSCENARIO2Parser#sIUnitSpecifier.
    def exitSIUnitSpecifier(self, ctx: OpenSCENARIO2Parser.SIUnitSpecifierContext):
        self.__current_scope = self.__current_scope.get_enclosing_scope()

    # Enter a parse tree produced by OpenSCENARIO2Parser#sIBaseExponentList.
    def enterSIBaseExponentList(
        self, ctx: OpenSCENARIO2Parser.SIBaseExponentListContext
    ):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#sIBaseExponentList.
    def exitSIBaseExponentList(
        self, ctx: OpenSCENARIO2Parser.SIBaseExponentListContext
    ):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#sIBaseExponent.
    def enterSIBaseExponent(self, ctx: OpenSCENARIO2Parser.SIBaseExponentContext):
        self.__node_stack.append(self.__cur_node)
        unit_name = ctx.Identifier().getText()
        value = ctx.integerLiteral().getText()

        si_base_exponent = SiExpSymbol(unit_name, value, self.__current_scope)
        self.__current_scope.define(si_base_exponent, ctx.start)
        self.__current_scope = si_base_exponent

        node = ast_node.SIBaseExponent(unit_name)
        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(self.__current_scope)

        self.__cur_node.set_children(node)
        self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#sIBaseExponent.
    def exitSIBaseExponent(self, ctx: OpenSCENARIO2Parser.SIBaseExponentContext):
        self.__cur_node = self.__node_stack.pop()
        self.__current_scope = self.__current_scope.get_enclosing_scope()

    # Enter a parse tree produced by OpenSCENARIO2Parser#sIFactor.
    def enterSIFactor(self, ctx: OpenSCENARIO2Parser.SIFactorContext):
        self.__node_stack.append(self.__cur_node)
        unit_name = "factor"
        node = ast_node.SIBaseExponent(unit_name)

        factor_value = None
        if ctx.FloatLiteral():
            float_value = ctx.FloatLiteral().getText()
            value_node = ast_node.FloatLiteral(float_value)
            node.set_children(value_node)
            factor_value = float_value
        elif ctx.integerLiteral():
            factor_value = ctx.integerLiteral().getText()

        si_base_exponent = SiExpSymbol(unit_name, factor_value, self.__current_scope)
        self.__current_scope.define(si_base_exponent, ctx.start)
        self.__current_scope = si_base_exponent

        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(self.__current_scope)

        self.__cur_node.set_children(node)
        self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#sIFactor.
    def exitSIFactor(self, ctx: OpenSCENARIO2Parser.SIFactorContext):
        self.__cur_node = self.__node_stack.pop()
        self.__current_scope = self.__current_scope.get_enclosing_scope()

    # Enter a parse tree produced by OpenSCENARIO2Parser#sIOffset.
    def enterSIOffset(self, ctx: OpenSCENARIO2Parser.SIOffsetContext):
        self.__node_stack.append(self.__cur_node)
        unit_name = "offset"
        node = ast_node.SIBaseExponent(unit_name)

        offset_value = None
        if ctx.FloatLiteral():
            float_value = ctx.FloatLiteral().getText()
            value_node = ast_node.FloatLiteral(float_value)
            value_node.set_loc(ctx.start.line, ctx.start.column)
            value_node.set_scope(self.__current_scope)
            node.set_children(value_node)
            offset_value = float_value
        elif ctx.integerLiteral():
            offset_value = ctx.integerLiteral().getText()

        si_base_exponent = SiExpSymbol(unit_name, offset_value, self.__current_scope)
        self.__current_scope.define(si_base_exponent, ctx.start)
        self.__current_scope = si_base_exponent

        if ctx.FloatLiteral():
            float_value = ctx.FloatLiteral().getText()
            string_symbol = FloatSymbol(self.__current_scope, float_value)
            self.__current_scope.define(string_symbol, ctx.start)

        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(self.__current_scope)

        self.__cur_node.set_children(node)
        self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#sIOffset.
    def exitSIOffset(self, ctx: OpenSCENARIO2Parser.SIOffsetContext):
        self.__cur_node = self.__node_stack.pop()
        self.__current_scope = self.__current_scope.get_enclosing_scope()

    # Enter a parse tree produced by OpenSCENARIO2Parser#enumDeclaration.
    def enterEnumDeclaration(self, ctx: OpenSCENARIO2Parser.EnumDeclarationContext):
        self.__node_stack.append(self.__cur_node)
        enum_name = ctx.enumName().getText()

        enum = EnumSymbol(enum_name, self.__current_scope)
        self.__current_scope.define(enum, ctx.start)
        self.__current_scope = enum

        node = ast_node.EnumDeclaration(enum_name)
        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(self.__current_scope)

        self.__cur_node.set_children(node)
        self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#enumDeclaration.
    def exitEnumDeclaration(self, ctx: OpenSCENARIO2Parser.EnumDeclarationContext):
        self.__cur_node = self.__node_stack.pop()
        self.__current_scope = self.__current_scope.get_enclosing_scope()

    # Enter a parse tree produced by OpenSCENARIO2Parser#enumMemberDecl.
    def enterEnumMemberDecl(self, ctx: OpenSCENARIO2Parser.EnumMemberDeclContext):
        # Here, it is assumed that the value of enum member is monotonically increasing and the step is 1
        # TODO The above assumptions do not necessarily fit the design of OSC2
        self.__node_stack.append(self.__cur_node)
        member_name = ctx.enumMemberName().getText()

        member_value = None
        if ctx.enumMemberValue():
            if ctx.enumMemberValue().UintLiteral():
                member_value = int(ctx.enumMemberValue().UintLiteral().getText())
            elif ctx.enumMemberValue().HexUintLiteral():
                member_value = int(ctx.enumMemberValue().HexUintLiteral().getText(), 16)
            else:
                pass

        if member_value is None:
            member_value = self.__current_scope.last_index + 1

        enum_member = EnumMemberSymbol(member_name, self.__current_scope, member_value)
        self.__current_scope.define(enum_member, ctx.start)

        # The AST enumeration value is the value stored in the symbol table
        node = ast_node.EnumMemberDecl(member_name, member_value)
        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(self.__current_scope)

        self.__cur_node.set_children(node)
        self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#enumMemberDecl.
    def exitEnumMemberDecl(self, ctx: OpenSCENARIO2Parser.EnumMemberDeclContext):
        self.__cur_node = self.__node_stack.pop()

    # Enter a parse tree produced by OpenSCENARIO2Parser#enumMemberValue.
    def enterEnumMemberValue(self, ctx: OpenSCENARIO2Parser.EnumMemberValueContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#enumMemberValue.
    def exitEnumMemberValue(self, ctx: OpenSCENARIO2Parser.EnumMemberValueContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#enumName.
    def enterEnumName(self, ctx: OpenSCENARIO2Parser.EnumNameContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#enumName.
    def exitEnumName(self, ctx: OpenSCENARIO2Parser.EnumNameContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#enumMemberName.
    def enterEnumMemberName(self, ctx: OpenSCENARIO2Parser.EnumMemberNameContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#enumMemberName.
    def exitEnumMemberName(self, ctx: OpenSCENARIO2Parser.EnumMemberNameContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#enumValueReference.
    def enterEnumValueReference(
        self, ctx: OpenSCENARIO2Parser.EnumValueReferenceContext
    ):
        self.__node_stack.append(self.__cur_node)
        enum_name = None
        if ctx.enumName():
            enum_name = ctx.enumName().getText()

        # Search the scope of enum_name
        scope = self.__current_scope.resolve(enum_name)
        enum_member_name = ctx.enumMemberName().getText()

        if scope and isinstance(scope, EnumSymbol):
            if scope.symbols.get(enum_member_name):
                enum_value_reference = EnumValueRefSymbol(
                    enum_name,
                    enum_member_name,
                    scope.symbols[enum_member_name].elems_index,
                    scope,
                )
                self.__current_scope.define(enum_value_reference, ctx.start)
            else:
                msg = "Enum member " + enum_member_name + " not found!"
                LOG_ERROR(msg, ctx.start)
        else:
            msg = "Enum " + enum_name + " not found!"
            LOG_ERROR(msg, ctx.start)

        node = ast_node.EnumValueReference(enum_name, enum_member_name)
        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(self.__current_scope)

        self.__cur_node.set_children(node)
        self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#enumValueReference.
    def exitEnumValueReference(
        self, ctx: OpenSCENARIO2Parser.EnumValueReferenceContext
    ):
        self.__cur_node = self.__node_stack.pop()

    # Enter a parse tree produced by OpenSCENARIO2Parser#inheritsCondition.
    def enterInheritsCondition(self, ctx: OpenSCENARIO2Parser.InheritsConditionContext):
        self.__node_stack.append(self.__cur_node)
        field_name = ctx.fieldName().getText()

        inherits_condition = InheritsConditionSymbol(field_name, self.__current_scope)
        self.__current_scope.define(inherits_condition, ctx.start)
        self.__current_scope = inherits_condition

        # Since there is no function to process bool_literal,
        # so we manually add a bool literal node to ast.
        bool_literal_node = None
        if ctx.BoolLiteral():
            bool_literal = ctx.BoolLiteral().getText()
            bool_literal_node = ast_node.BoolLiteral(bool_literal)
            bool_symbol = BoolSymbol(self.__current_scope, bool_literal)
            self.__current_scope.define(bool_symbol, ctx.start)

        node = ast_node.InheritsCondition(field_name, bool_literal_node)
        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(self.__current_scope)

        self.__cur_node.set_children(node)
        self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#inheritsCondition.
    def exitInheritsCondition(self, ctx: OpenSCENARIO2Parser.InheritsConditionContext):
        self.__cur_node = self.__node_stack.pop()
        self.__current_scope = self.__current_scope.get_enclosing_scope()

    # Enter a parse tree produced by OpenSCENARIO2Parser#structDeclaration.
    def enterStructDeclaration(self, ctx: OpenSCENARIO2Parser.StructDeclarationContext):
        self.__node_stack.append(self.__cur_node)
        struct_name = ctx.structName().getText()

        struct = StructSymbol(struct_name, self.__current_scope)
        self.__current_scope.define(struct, ctx.start)
        self.__current_scope = struct

        node = ast_node.StructDeclaration(struct_name)
        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(self.__current_scope)

        self.__cur_node.set_children(node)
        self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#structDeclaration.
    def exitStructDeclaration(self, ctx: OpenSCENARIO2Parser.StructDeclarationContext):
        self.__cur_node = self.__node_stack.pop()
        self.__current_scope = self.__current_scope.get_enclosing_scope()

    # Enter a parse tree produced by OpenSCENARIO2Parser#structInherts.
    def enterStructInherts(self, ctx: OpenSCENARIO2Parser.StructInhertsContext):
        self.__node_stack.append(self.__cur_node)
        struct_name = ctx.structName().getText()

        # Finds the scope of the inherited actor_name
        scope = self.__current_scope.resolve(struct_name)

        if scope and isinstance(scope, StructSymbol):
            pass
        else:
            msg = "inherits " + struct_name + " is not defined!"
            LOG_ERROR(msg, ctx.start)

        struct_inherts = StructInhertsSymbol(struct_name, self.__current_scope, scope)
        self.__current_scope.define(struct_inherts, ctx.start)

        node = ast_node.StructInherts(struct_name)
        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(self.__current_scope)

        self.__cur_node.set_children(node)
        self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#structInherts.
    def exitStructInherts(self, ctx: OpenSCENARIO2Parser.StructInhertsContext):
        self.__cur_node = self.__node_stack.pop()

    # Enter a parse tree produced by OpenSCENARIO2Parser#structMemberDecl.
    def enterStructMemberDecl(self, ctx: OpenSCENARIO2Parser.StructMemberDeclContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#structMemberDecl.
    def exitStructMemberDecl(self, ctx: OpenSCENARIO2Parser.StructMemberDeclContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#fieldName.
    def enterFieldName(self, ctx: OpenSCENARIO2Parser.FieldNameContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#fieldName.
    def exitFieldName(self, ctx: OpenSCENARIO2Parser.FieldNameContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#structName.
    def enterStructName(self, ctx: OpenSCENARIO2Parser.StructNameContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#structName.
    def exitStructName(self, ctx: OpenSCENARIO2Parser.StructNameContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#actorDeclaration.
    def enterActorDeclaration(self, ctx: OpenSCENARIO2Parser.ActorDeclarationContext):
        self.__node_stack.append(self.__cur_node)
        actor_name = ctx.actorName().getText()

        actor = ActorSymbol(actor_name, self.__current_scope)
        self.__current_scope.define(actor, ctx.start)
        self.__current_scope = actor

        node = ast_node.ActorDeclaration(actor_name)
        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(self.__current_scope)

        self.__cur_node.set_children(node)
        self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#actorDeclaration.
    def exitActorDeclaration(self, ctx: OpenSCENARIO2Parser.ActorDeclarationContext):
        self.__cur_node = self.__node_stack.pop()
        self.__current_scope = self.__current_scope.get_enclosing_scope()

    # Enter a parse tree produced by OpenSCENARIO2Parser#actorInherts.
    def enterActorInherts(self, ctx: OpenSCENARIO2Parser.ActorInhertsContext):
        self.__node_stack.append(self.__cur_node)
        actor_name = ctx.actorName().getText()

        # Finds the scope of the inherited actor_name
        scope = self.__current_scope.resolve(actor_name)

        if scope and isinstance(scope, ActorSymbol):
            pass
        else:
            scope = None
            msg = "inherits " + actor_name + " is not defined!"
            LOG_ERROR(msg, ctx.start)

        actor_inherits = ActorInhertsSymbol(actor_name, self.__current_scope, scope)
        self.__current_scope.define(actor_inherits, ctx.start)

        node = ast_node.ActorInherts(actor_name)
        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(self.__current_scope)

        self.__cur_node.set_children(node)
        self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#actorInherts.
    def exitActorInherts(self, ctx: OpenSCENARIO2Parser.ActorInhertsContext):
        self.__cur_node = self.__node_stack.pop()

    # Enter a parse tree produced by OpenSCENARIO2Parser#actorMemberDecl.
    def enterActorMemberDecl(self, ctx: OpenSCENARIO2Parser.ActorMemberDeclContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#actorMemberDecl.
    def exitActorMemberDecl(self, ctx: OpenSCENARIO2Parser.ActorMemberDeclContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#actorName.
    def enterActorName(self, ctx: OpenSCENARIO2Parser.ActorNameContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#actorName.
    def exitActorName(self, ctx: OpenSCENARIO2Parser.ActorNameContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#scenarioDeclaration.
    def enterScenarioDeclaration(
        self, ctx: OpenSCENARIO2Parser.ScenarioDeclarationContext
    ):
        self.__node_stack.append(self.__cur_node)
        qualified_behavior_name = ctx.qualifiedBehaviorName().getText()

        scenario_name = QualifiedBehaviorSymbol(
            qualified_behavior_name, self.__current_scope
        )
        scenario_name.is_qualified_behavior_name_valid(ctx.start)
        scenario = ScenarioSymbol(scenario_name)
        self.__current_scope.define(scenario, ctx.start)

        node = ast_node.ScenarioDeclaration(qualified_behavior_name)
        node.set_loc(ctx.start.line, ctx.start.column)

        scenario.declaration_address = node
        self.__current_scope = scenario
        node.set_scope(self.__current_scope)

        self.__cur_node.set_children(node)
        self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#scenarioDeclaration.
    def exitScenarioDeclaration(
        self, ctx: OpenSCENARIO2Parser.ScenarioDeclarationContext
    ):
        self.__cur_node = self.__node_stack.pop()
        self.__current_scope = self.__current_scope.get_enclosing_scope()

    # Enter a parse tree produced by OpenSCENARIO2Parser#scenarioInherts.
    def enterScenarioInherts(self, ctx: OpenSCENARIO2Parser.ScenarioInhertsContext):
        self.__node_stack.append(self.__cur_node)
        qualified_behavior_name = ctx.qualifiedBehaviorName().getText()

        # Searches the scope of the inheritance scenario_name
        scope = self.__current_scope.resolve(qualified_behavior_name)

        if scope and isinstance(scope, ScenarioSymbol):
            scenario_name = QualifiedBehaviorSymbol(
                qualified_behavior_name, self.__current_scope
            )
            scenario_name.is_qualified_behavior_name_valid(ctx.start)
            scenario_inherts = ScenarioInhertsSymbol(scenario_name)

            self.__current_scope.define(scenario_inherts, ctx.start)
            self.__current_scope = scenario_inherts
        else:
            msg = "inherits " + qualified_behavior_name + " is not defined!"
            LOG_ERROR(msg, ctx.start)

        node = ast_node.ScenarioInherts(qualified_behavior_name)
        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(self.__current_scope)

        self.__cur_node.set_children(node)
        self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#scenarioInherts.
    def exitScenarioInherts(self, ctx: OpenSCENARIO2Parser.ScenarioInhertsContext):
        self.__cur_node = self.__node_stack.pop()
        self.__current_scope = self.__current_scope.get_enclosing_scope()

    # Enter a parse tree produced by OpenSCENARIO2Parser#scenarioMemberDecl.
    def enterScenarioMemberDecl(
        self, ctx: OpenSCENARIO2Parser.ScenarioMemberDeclContext
    ):
        self.__node_stack.append(self.__cur_node)

    # Exit a parse tree produced by OpenSCENARIO2Parser#scenarioMemberDecl.
    def exitScenarioMemberDecl(
        self, ctx: OpenSCENARIO2Parser.ScenarioMemberDeclContext
    ):
        self.__cur_node = self.__node_stack.pop()

    # Enter a parse tree produced by OpenSCENARIO2Parser#qualifiedBehaviorName.
    def enterQualifiedBehaviorName(
        self, ctx: OpenSCENARIO2Parser.QualifiedBehaviorNameContext
    ):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#qualifiedBehaviorName.
    def exitQualifiedBehaviorName(
        self, ctx: OpenSCENARIO2Parser.QualifiedBehaviorNameContext
    ):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#behaviorName.
    def enterBehaviorName(self, ctx: OpenSCENARIO2Parser.BehaviorNameContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#behaviorName.
    def exitBehaviorName(self, ctx: OpenSCENARIO2Parser.BehaviorNameContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#actionDeclaration.
    def enterActionDeclaration(self, ctx: OpenSCENARIO2Parser.ActionDeclarationContext):
        self.__node_stack.append(self.__cur_node)
        qualified_behavior_name = ctx.qualifiedBehaviorName().getText()

        action_name = QualifiedBehaviorSymbol(
            qualified_behavior_name, self.__current_scope
        )
        action_name.is_qualified_behavior_name_valid(ctx.start)
        action = ActionSymbol(action_name)
        self.__current_scope.define(action, ctx.start)
        self.__current_scope = action

        node = ast_node.ActionDeclaration(qualified_behavior_name)
        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(self.__current_scope)

        self.__cur_node.set_children(node)
        self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#actionDeclaration.
    def exitActionDeclaration(self, ctx: OpenSCENARIO2Parser.ActionDeclarationContext):
        self.__cur_node = self.__node_stack.pop()
        self.__current_scope = self.__current_scope.get_enclosing_scope()

    # Enter a parse tree produced by OpenSCENARIO2Parser#actionInherts.
    def enterActionInherts(self, ctx: OpenSCENARIO2Parser.ActionInhertsContext):
        self.__node_stack.append(self.__cur_node)
        qualified_behavior_name = ctx.qualifiedBehaviorName().getText()

        # Finds the scope of the inherited action_name
        scope = self.__current_scope.resolve(qualified_behavior_name)

        if scope and isinstance(scope, ActionSymbol):
            pass
        else:
            msg = "inherits " + qualified_behavior_name + " is not defined!"
            LOG_ERROR(msg, ctx.start)

        action_name = QualifiedBehaviorSymbol(
            qualified_behavior_name, self.__current_scope
        )
        action_name.is_qualified_behavior_name_valid(ctx.start)
        action_inherits = ActionInhertsSymbol(action_name, scope)
        self.__current_scope.define(action_inherits, ctx.start)

        node = ast_node.ActionInherts(qualified_behavior_name)
        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(self.__current_scope)

        self.__cur_node.set_children(node)
        self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#actionInherts.
    def exitActionInherts(self, ctx: OpenSCENARIO2Parser.ActionInhertsContext):
        self.__cur_node = self.__node_stack.pop()

    # Enter a parse tree produced by OpenSCENARIO2Parser#modifierDeclaration.
    def enterModifierDeclaration(
        self, ctx: OpenSCENARIO2Parser.ModifierDeclarationContext
    ):
        self.__node_stack.append(self.__cur_node)
        actor_name = None
        if ctx.actorName():
            actor_name = ctx.actorName().getText()
            if self.__current_scope.resolve(actor_name):
                pass
            else:
                msg = "Actor: " + actor_name + " is not defined!"
                LOG_ERROR(msg, ctx.start)

        modifier_name = ctx.modifierName().getText()

        modifier = ModifierSymbol(modifier_name, self.__current_scope)
        self.__current_scope.define(modifier, ctx.start)
        self.__current_scope = modifier

        node = ast_node.ModifierDeclaration(actor_name, modifier_name)
        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(self.__current_scope)

        self.__cur_node.set_children(node)
        self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#modifierDeclaration.
    def exitModifierDeclaration(
        self, ctx: OpenSCENARIO2Parser.ModifierDeclarationContext
    ):
        self.__cur_node = self.__node_stack.pop()
        self.__current_scope = self.__current_scope.get_enclosing_scope()

    # Enter a parse tree produced by OpenSCENARIO2Parser#modifierName.
    def enterModifierName(self, ctx: OpenSCENARIO2Parser.ModifierNameContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#modifierName.
    def exitModifierName(self, ctx: OpenSCENARIO2Parser.ModifierNameContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#typeExtension.
    def enterTypeExtension(self, ctx: OpenSCENARIO2Parser.TypeExtensionContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#typeExtension.
    def exitTypeExtension(self, ctx: OpenSCENARIO2Parser.TypeExtensionContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#enumTypeExtension.
    def enterEnumTypeExtension(self, ctx: OpenSCENARIO2Parser.EnumTypeExtensionContext):
        # TODO Here use the symbol table to query the value of the last element of the extended enum
        #  and set self.__enum_member_idx
        # Finds the same enum in the current scope (global scope)
        self.__node_stack.append(self.__cur_node)
        enum_name = ctx.enumName().getText()

        if enum_name in self.__current_scope.symbols and isinstance(
            self.__current_scope.symbols[enum_name], EnumSymbol
        ):
            self.__current_scope = self.__current_scope.symbols[enum_name]
        else:
            msg = enum_name + " is not defined!"
            LOG_ERROR(msg, ctx.start)

        node = ast_node.EnumTypeExtension(enum_name)
        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(self.__current_scope)

        self.__cur_node.set_children(node)
        self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#enumTypeExtension.
    def exitEnumTypeExtension(self, ctx: OpenSCENARIO2Parser.EnumTypeExtensionContext):
        self.__cur_node = self.__node_stack.pop()
        self.__current_scope = self.__current_scope.get_enclosing_scope()

    # Enter a parse tree produced by OpenSCENARIO2Parser#structuredTypeExtension.
    def enterStructuredTypeExtension(
        self, ctx: OpenSCENARIO2Parser.StructuredTypeExtensionContext
    ):
        self.__node_stack.append(self.__cur_node)

        type_name = None
        if ctx.extendableTypeName().typeName():
            type_name = ctx.extendableTypeName().typeName().getText()
            # Even if a symbol table with the corresponding name is found,
            # it is necessary to determine whether the extended symbol table matches the original type
            if type_name in self.__current_scope.symbols:
                for extend_member in ctx.extensionMemberDecl():
                    if extend_member.structMemberDecl() and isinstance(
                        self.__current_scope.symbols[type_name], StructSymbol
                    ):
                        self.__current_scope = self.__current_scope.symbols[type_name]
                    elif extend_member.actorMemberDecl and isinstance(
                        self.__current_scope.symbols[type_name], ActorSymbol
                    ):
                        self.__current_scope = self.__current_scope.symbols[type_name]
                    elif extend_member.scenarioMemberDecl and isinstance(
                        self.__current_scope.symbols[type_name], ScenarioSymbol
                    ):
                        self.__current_scope = self.__current_scope.symbols[type_name]
                    elif extend_member.behaviorSpecification:
                        # TODO
                        msg = "I haven't written the code yet"
                        LOG_ERROR(msg, ctx.start)
                    else:
                        msg = type_name + " is not defined!"
                        LOG_ERROR(msg, ctx.start)
            else:
                msg = type_name + " is not defined!"
                LOG_ERROR(msg, ctx.start)

        # The two ifs here are syntactically mutually exclusive
        qualified_behavior_name = None
        if ctx.extendableTypeName().qualifiedBehaviorName():
            qualified_behavior_name = (
                ctx.extendableTypeName().qualifiedBehaviorName().getText()
            )
            if qualified_behavior_name in self.__current_scope.symbols:
                for extend_member in ctx.extensionMemberDecl():
                    if extend_member.scenarioMemberDecl and isinstance(
                        self.__current_scope.symbols[qualified_behavior_name],
                        ScenarioSymbol,
                    ):
                        self.__current_scope = self.__current_scope.symbols[
                            qualified_behavior_name
                        ]
                    elif extend_member.behaviorSpecification:
                        # TODO
                        msg = "not implemented"
                        LOG_ERROR(msg, ctx.start)
                    else:
                        msg = qualified_behavior_name + " is not defined!"
                        LOG_ERROR(msg, ctx.start)
            else:
                msg = qualified_behavior_name + " is Not defined!"
                LOG_ERROR(msg, ctx.start)

        node = ast_node.StructuredTypeExtension(type_name, qualified_behavior_name)
        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(self.__current_scope)

        self.__cur_node.set_children(node)
        self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#structuredTypeExtension.
    def exitStructuredTypeExtension(
        self, ctx: OpenSCENARIO2Parser.StructuredTypeExtensionContext
    ):
        self.__cur_node = self.__node_stack.pop()
        self.__current_scope = self.__current_scope.get_enclosing_scope()

    # Enter a parse tree produced by OpenSCENARIO2Parser#extendableTypeName.
    def enterExtendableTypeName(
        self, ctx: OpenSCENARIO2Parser.ExtendableTypeNameContext
    ):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#extendableTypeName.
    def exitExtendableTypeName(
        self, ctx: OpenSCENARIO2Parser.ExtendableTypeNameContext
    ):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#extensionMemberDecl.
    def enterExtensionMemberDecl(
        self, ctx: OpenSCENARIO2Parser.ExtensionMemberDeclContext
    ):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#extensionMemberDecl.
    def exitExtensionMemberDecl(
        self, ctx: OpenSCENARIO2Parser.ExtensionMemberDeclContext
    ):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#globalParameterDeclaration.
    def enterGlobalParameterDeclaration(
        self, ctx: OpenSCENARIO2Parser.GlobalParameterDeclarationContext
    ):
        defaultValue = None
        if ctx.defaultValue():
            defaultValue = ctx.defaultValue().getText()
        field_type = ctx.typeDeclarator().getText()
        self.__node_stack.append(self.__cur_node)
        field_name = []
        multi_field_name = ""
        for fn in ctx.fieldName():
            name = fn.Identifier().getText()
            if name in field_name:
                msg = "Can not define same param in same scope!"
                LOG_ERROR(msg, ctx.start)
            field_name.append(name)
            multi_field_name = multi_field_name_append(multi_field_name, name)

        parameter = ParameterSymbol(
            multi_field_name, self.__current_scope, field_type, defaultValue
        )
        self.__current_scope.define(parameter, ctx.start)
        self.__current_scope = parameter

        node = ast_node.GlobalParameterDeclaration(field_name, field_type)
        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(self.__current_scope)

        self.__cur_node.set_children(node)
        self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#globalParameterDeclaration.
    def exitGlobalParameterDeclaration(
        self, ctx: OpenSCENARIO2Parser.GlobalParameterDeclarationContext
    ):
        self.__cur_node = self.__node_stack.pop()
        self.__current_scope = self.__current_scope.get_enclosing_scope()

    # Enter a parse tree produced by OpenSCENARIO2Parser#typeDeclarator.
    def enterTypeDeclarator(self, ctx: OpenSCENARIO2Parser.TypeDeclaratorContext):
        self.__node_stack.append(self.__cur_node)
        type_name = None
        if ctx.nonAggregateTypeDeclarator():
            type_name = ctx.nonAggregateTypeDeclarator().getText()
        elif ctx.aggregateTypeDeclarator():
            type_name = ctx.aggregateTypeDeclarator().getText()

        node = ast_node.Type(type_name)
        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(self.__current_scope)

        self.__cur_node.set_children(node)
        self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#typeDeclarator.
    def exitTypeDeclarator(self, ctx: OpenSCENARIO2Parser.TypeDeclaratorContext):
        self.__cur_node = self.__node_stack.pop()

    # Enter a parse tree produced by OpenSCENARIO2Parser#nonAggregateTypeDeclarator.
    def enterNonAggregateTypeDeclarator(
        self, ctx: OpenSCENARIO2Parser.NonAggregateTypeDeclaratorContext
    ):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#nonAggregateTypeDeclarator.
    def exitNonAggregateTypeDeclarator(
        self, ctx: OpenSCENARIO2Parser.NonAggregateTypeDeclaratorContext
    ):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#aggregateTypeDeclarator.
    def enterAggregateTypeDeclarator(
        self, ctx: OpenSCENARIO2Parser.AggregateTypeDeclaratorContext
    ):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#aggregateTypeDeclarator.
    def exitAggregateTypeDeclarator(
        self, ctx: OpenSCENARIO2Parser.AggregateTypeDeclaratorContext
    ):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#listTypeDeclarator.
    def enterListTypeDeclarator(
        self, ctx: OpenSCENARIO2Parser.ListTypeDeclaratorContext
    ):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#listTypeDeclarator.
    def exitListTypeDeclarator(
        self, ctx: OpenSCENARIO2Parser.ListTypeDeclaratorContext
    ):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#primitiveType.
    def enterPrimitiveType(self, ctx: OpenSCENARIO2Parser.PrimitiveTypeContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#primitiveType.
    def exitPrimitiveType(self, ctx: OpenSCENARIO2Parser.PrimitiveTypeContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#typeName.
    def enterTypeName(self, ctx: OpenSCENARIO2Parser.TypeNameContext):
        name = ctx.Identifier().getText()
        # Find the scope of type_name
        scope = self.__current_scope.resolve(name)

        if scope:
            pass
        else:
            msg = "Type name: " + name + " is not defined!"
            LOG_ERROR(msg, ctx.start)

    # Exit a parse tree produced by OpenSCENARIO2Parser#typeName.
    def exitTypeName(self, ctx: OpenSCENARIO2Parser.TypeNameContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#eventDeclaration.
    def enterEventDeclaration(self, ctx: OpenSCENARIO2Parser.EventDeclarationContext):
        self.__node_stack.append(self.__cur_node)
        event_name = ctx.eventName().getText()

        node = ast_node.EventDeclaration(event_name)
        node.set_loc(ctx.start.line, ctx.start.column)

        event = EventSymbol(event_name, self.__current_scope)
        self.__current_scope.define(event, ctx.start)
        self.__current_scope = event

        self.__current_scope.declaration_address = node

        node.set_scope(self.__current_scope)

        self.__cur_node.set_children(node)
        self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#eventDeclaration.
    def exitEventDeclaration(self, ctx: OpenSCENARIO2Parser.EventDeclarationContext):
        self.__cur_node = self.__node_stack.pop()
        self.__current_scope = self.__current_scope.get_enclosing_scope()

    # Enter a parse tree produced by OpenSCENARIO2Parser#eventSpecification.
    def enterEventSpecification(
        self, ctx: OpenSCENARIO2Parser.EventSpecificationContext
    ):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#eventSpecification.
    def exitEventSpecification(
        self, ctx: OpenSCENARIO2Parser.EventSpecificationContext
    ):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#eventReference.
    def enterEventReference(self, ctx: OpenSCENARIO2Parser.EventReferenceContext):
        self.__node_stack.append(self.__cur_node)
        event_path = ctx.eventPath().getText()
        node = ast_node.EventReference(event_path)
        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(self.__current_scope)

        self.__cur_node.set_children(node)
        self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#eventReference.
    def exitEventReference(self, ctx: OpenSCENARIO2Parser.EventReferenceContext):
        self.__cur_node = self.__node_stack.pop()

    # Enter a parse tree produced by OpenSCENARIO2Parser#eventFieldDecl.
    def enterEventFieldDecl(self, ctx: OpenSCENARIO2Parser.EventFieldDeclContext):
        self.__node_stack.append(self.__cur_node)
        event_field_name = ctx.eventFieldName().getText()
        node = ast_node.EventFieldDecl(event_field_name)
        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(self.__current_scope)

        self.__cur_node.set_children(node)
        self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#eventFieldDecl.
    def exitEventFieldDecl(self, ctx: OpenSCENARIO2Parser.EventFieldDeclContext):
        self.__cur_node = self.__node_stack.pop()

    # Enter a parse tree produced by OpenSCENARIO2Parser#eventFieldName.
    def enterEventFieldName(self, ctx: OpenSCENARIO2Parser.EventFieldNameContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#eventFieldName.
    def exitEventFieldName(self, ctx: OpenSCENARIO2Parser.EventFieldNameContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#eventName.
    def enterEventName(self, ctx: OpenSCENARIO2Parser.EventNameContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#eventName.
    def exitEventName(self, ctx: OpenSCENARIO2Parser.EventNameContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#eventPath.
    def enterEventPath(self, ctx: OpenSCENARIO2Parser.EventPathContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#eventPath.
    def exitEventPath(self, ctx: OpenSCENARIO2Parser.EventPathContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#eventCondition.
    def enterEventCondition(self, ctx: OpenSCENARIO2Parser.EventConditionContext):
        self.__node_stack.append(self.__cur_node)
        node = ast_node.EventCondition()
        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(self.__current_scope)

        self.__cur_node.set_children(node)
        self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#eventCondition.
    def exitEventCondition(self, ctx: OpenSCENARIO2Parser.EventConditionContext):
        self.__cur_node = self.__node_stack.pop()

    # Enter a parse tree produced by OpenSCENARIO2Parser#riseExpression.
    def enterRiseExpression(self, ctx: OpenSCENARIO2Parser.RiseExpressionContext):
        self.__node_stack.append(self.__cur_node)
        node = ast_node.RiseExpression()
        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(self.__current_scope)

        self.__cur_node.set_children(node)
        self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#riseExpression.
    def exitRiseExpression(self, ctx: OpenSCENARIO2Parser.RiseExpressionContext):
        self.__cur_node = self.__node_stack.pop()

    # Enter a parse tree produced by OpenSCENARIO2Parser#fallExpression.
    def enterFallExpression(self, ctx: OpenSCENARIO2Parser.FallExpressionContext):
        self.__node_stack.append(self.__cur_node)
        node = ast_node.FallExpression()
        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(self.__current_scope)

        self.__cur_node.set_children(node)
        self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#fallExpression.
    def exitFallExpression(self, ctx: OpenSCENARIO2Parser.FallExpressionContext):
        self.__cur_node = self.__node_stack.pop()

    # Enter a parse tree produced by OpenSCENARIO2Parser#elapsedExpression.
    def enterElapsedExpression(self, ctx: OpenSCENARIO2Parser.ElapsedExpressionContext):
        self.__node_stack.append(self.__cur_node)
        node = ast_node.ElapsedExpression()
        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(self.__current_scope)

        self.__cur_node.set_children(node)
        self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#elapsedExpression.
    def exitElapsedExpression(self, ctx: OpenSCENARIO2Parser.ElapsedExpressionContext):
        self.__cur_node = self.__node_stack.pop()

    # Enter a parse tree produced by OpenSCENARIO2Parser#everyExpression.
    def enterEveryExpression(self, ctx: OpenSCENARIO2Parser.EveryExpressionContext):
        self.__node_stack.append(self.__cur_node)
        node = ast_node.EveryExpression()
        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(self.__current_scope)

        self.__cur_node.set_children(node)
        self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#everyExpression.
    def exitEveryExpression(self, ctx: OpenSCENARIO2Parser.EveryExpressionContext):
        self.__cur_node = self.__node_stack.pop()

    # Enter a parse tree produced by OpenSCENARIO2Parser#boolExpression.
    def enterBoolExpression(self, ctx: OpenSCENARIO2Parser.BoolExpressionContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#boolExpression.
    def exitBoolExpression(self, ctx: OpenSCENARIO2Parser.BoolExpressionContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#durationExpression.
    def enterDurationExpression(
        self, ctx: OpenSCENARIO2Parser.DurationExpressionContext
    ):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#durationExpression.
    def exitDurationExpression(
        self, ctx: OpenSCENARIO2Parser.DurationExpressionContext
    ):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#fieldDeclaration.
    def enterFieldDeclaration(self, ctx: OpenSCENARIO2Parser.FieldDeclarationContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#fieldDeclaration.
    def exitFieldDeclaration(self, ctx: OpenSCENARIO2Parser.FieldDeclarationContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#parameterDeclaration.
    def enterParameterDeclaration(
        self, ctx: OpenSCENARIO2Parser.ParameterDeclarationContext
    ):
        defaultValue = None
        if ctx.defaultValue():
            defaultValue = ctx.defaultValue().getText()
        field_type = ctx.typeDeclarator().getText()
        self.__node_stack.append(self.__cur_node)
        field_name = []

        multi_field_name = ""
        for fn in ctx.fieldName():
            name = fn.Identifier().getText()
            if name in field_name:
                msg = "Can not define same param in same scope!"
                LOG_ERROR(msg, ctx.start)
            field_name.append(name)
            multi_field_name = multi_field_name_append(multi_field_name, name)

        parameter = ParameterSymbol(
            multi_field_name, self.__current_scope, field_type, defaultValue
        )
        self.__current_scope.define(parameter, ctx.start)
        self.__current_scope = parameter

        node = ast_node.ParameterDeclaration(field_name, field_type)
        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(self.__current_scope)

        self.__cur_node.set_children(node)
        self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#parameterDeclaration.
    def exitParameterDeclaration(
        self, ctx: OpenSCENARIO2Parser.ParameterDeclarationContext
    ):
        self.__cur_node = self.__node_stack.pop()
        self.__current_scope = self.__current_scope.get_enclosing_scope()

    # Enter a parse tree produced by OpenSCENARIO2Parser#variableDeclaration.
    def enterVariableDeclaration(
        self, ctx: OpenSCENARIO2Parser.VariableDeclarationContext
    ):
        self.__node_stack.append(self.__cur_node)
        field_name = []
        defaultValue = None
        if ctx.sampleExpression():
            defaultValue = ctx.sampleExpression().getText()
        elif ctx.valueExp():
            defaultValue = ctx.valueExp().getText()

        multi_field_name = ""
        for fn in ctx.fieldName():
            name = fn.Identifier().getText()
            if name in field_name:
                msg = "Can not define same param in same scope!"
                LOG_ERROR(msg, ctx.start)
            field_name.append(name)
            multi_field_name = multi_field_name_append(multi_field_name, name)

        field_type = ctx.typeDeclarator().getText()

        variable = VariableSymbol(
            multi_field_name, self.__current_scope, field_type, defaultValue
        )
        self.__current_scope.define(variable, ctx.start)
        self.__current_scope = variable

        node = ast_node.VariableDeclaration(field_name, field_type)
        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(self.__current_scope)

        self.__cur_node.set_children(node)
        self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#variableDeclaration.
    def exitVariableDeclaration(
        self, ctx: OpenSCENARIO2Parser.VariableDeclarationContext
    ):
        self.__cur_node = self.__node_stack.pop()
        self.__current_scope = self.__current_scope.get_enclosing_scope()

    # Enter a parse tree produced by OpenSCENARIO2Parser#sampleExpression.
    def enterSampleExpression(self, ctx: OpenSCENARIO2Parser.SampleExpressionContext):
        self.__node_stack.append(self.__cur_node)
        node = ast_node.SampleExpression()
        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(self.__current_scope)

        self.__cur_node.set_children(node)
        self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#sampleExpression.
    def exitSampleExpression(self, ctx: OpenSCENARIO2Parser.SampleExpressionContext):
        self.__cur_node = self.__node_stack.pop()

    # Enter a parse tree produced by OpenSCENARIO2Parser#defaultValue.
    def enterDefaultValue(self, ctx: OpenSCENARIO2Parser.DefaultValueContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#defaultValue.
    def exitDefaultValue(self, ctx: OpenSCENARIO2Parser.DefaultValueContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#parameterWithDeclaration.
    def enterParameterWithDeclaration(
        self, ctx: OpenSCENARIO2Parser.ParameterWithDeclarationContext
    ):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#parameterWithDeclaration.
    def exitParameterWithDeclaration(
        self, ctx: OpenSCENARIO2Parser.ParameterWithDeclarationContext
    ):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#parameterWithMember.
    def enterParameterWithMember(
        self, ctx: OpenSCENARIO2Parser.ParameterWithMemberContext
    ):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#parameterWithMember.
    def exitParameterWithMember(
        self, ctx: OpenSCENARIO2Parser.ParameterWithMemberContext
    ):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#constraintDeclaration.
    def enterConstraintDeclaration(
        self, ctx: OpenSCENARIO2Parser.ConstraintDeclarationContext
    ):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#constraintDeclaration.
    def exitConstraintDeclaration(
        self, ctx: OpenSCENARIO2Parser.ConstraintDeclarationContext
    ):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#keepConstraintDeclaration.
    def enterKeepConstraintDeclaration(
        self, ctx: OpenSCENARIO2Parser.KeepConstraintDeclarationContext
    ):
        self.__node_stack.append(self.__cur_node)
        constraint_qualifier = None
        if ctx.constraintQualifier():
            constraint_qualifier = ctx.constraintQualifier().getText()

        keep_symbol = KeepScope(self.__current_scope, constraint_qualifier)
        self.__current_scope.define(keep_symbol, ctx.start)
        self.__current_scope = keep_symbol

        node = ast_node.KeepConstraintDeclaration(constraint_qualifier)
        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(self.__current_scope)

        self.__cur_node.set_children(node)
        self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#keepConstraintDeclaration.
    def exitKeepConstraintDeclaration(
        self, ctx: OpenSCENARIO2Parser.KeepConstraintDeclarationContext
    ):
        self.__cur_node = self.__node_stack.pop()
        self.__current_scope = self.__current_scope.get_enclosing_scope()

    # Enter a parse tree produced by OpenSCENARIO2Parser#constraintQualifier.
    def enterConstraintQualifier(
        self, ctx: OpenSCENARIO2Parser.ConstraintQualifierContext
    ):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#constraintQualifier.
    def exitConstraintQualifier(
        self, ctx: OpenSCENARIO2Parser.ConstraintQualifierContext
    ):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#constraintExpression.
    def enterConstraintExpression(
        self, ctx: OpenSCENARIO2Parser.ConstraintExpressionContext
    ):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#constraintExpression.
    def exitConstraintExpression(
        self, ctx: OpenSCENARIO2Parser.ConstraintExpressionContext
    ):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#removeDefaultDeclaration.
    def enterRemoveDefaultDeclaration(
        self, ctx: OpenSCENARIO2Parser.RemoveDefaultDeclarationContext
    ):
        self.__node_stack.append(self.__cur_node)
        node = ast_node.RemoveDefaultDeclaration()
        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(self.__current_scope)

        self.__cur_node.set_children(node)
        self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#removeDefaultDeclaration.
    def exitRemoveDefaultDeclaration(
        self, ctx: OpenSCENARIO2Parser.RemoveDefaultDeclarationContext
    ):
        self.__cur_node = self.__node_stack.pop()

    # Enter a parse tree produced by OpenSCENARIO2Parser#parameterReference.
    def enterParameterReference(
        self, ctx: OpenSCENARIO2Parser.ParameterReferenceContext
    ):
        self.__node_stack.append(self.__cur_node)
        field_name = None
        if ctx.fieldName():
            field_name = ctx.fieldName().getText()

        field_access = None
        if ctx.fieldAccess():
            field_access = ctx.fieldAccess().getText()

        node = ast_node.ParameterReference(field_name, field_access)
        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(self.__current_scope)

        self.__cur_node.set_children(node)
        self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#parameterReference.
    def exitParameterReference(
        self, ctx: OpenSCENARIO2Parser.ParameterReferenceContext
    ):
        self.__cur_node = self.__node_stack.pop()

    # Enter a parse tree produced by OpenSCENARIO2Parser#modifierInvocation.
    def enterModifierInvocation(
        self, ctx: OpenSCENARIO2Parser.ModifierInvocationContext
    ):
        self.__node_stack.append(self.__cur_node)
        modifier_name = ctx.modifierName().getText()

        actor = None
        if ctx.actorExpression():
            actor = ctx.actorExpression().getText()

        if ctx.behaviorExpression():
            actor = ctx.behaviorExpression().getText()

        scope = None
        if actor is not None:
            scope = self.__current_scope.resolve(actor)
            if scope:
                pass
            else:
                msg = actor + " is not defined!"
                LOG_ERROR(msg, ctx.start)

        if scope is None:
            scope = self.__current_scope

        node = ast_node.ModifierInvocation(actor, modifier_name)
        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(scope)

        self.__cur_node.set_children(node)
        self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#modifierInvocation.
    def exitModifierInvocation(
        self, ctx: OpenSCENARIO2Parser.ModifierInvocationContext
    ):
        self.__cur_node = self.__node_stack.pop()

    # Enter a parse tree produced by OpenSCENARIO2Parser#behaviorExpression.
    def enterBehaviorExpression(
        self, ctx: OpenSCENARIO2Parser.BehaviorExpressionContext
    ):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#behaviorExpression.
    def exitBehaviorExpression(
        self, ctx: OpenSCENARIO2Parser.BehaviorExpressionContext
    ):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#behaviorSpecification.
    def enterBehaviorSpecification(
        self, ctx: OpenSCENARIO2Parser.BehaviorSpecificationContext
    ):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#behaviorSpecification.
    def exitBehaviorSpecification(
        self, ctx: OpenSCENARIO2Parser.BehaviorSpecificationContext
    ):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#onDirective.
    def enterOnDirective(self, ctx: OpenSCENARIO2Parser.OnDirectiveContext):
        self.__node_stack.append(self.__cur_node)
        node = ast_node.OnDirective()
        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(self.__current_scope)

        self.__cur_node.set_children(node)
        self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#onDirective.
    def exitOnDirective(self, ctx: OpenSCENARIO2Parser.OnDirectiveContext):
        self.__cur_node = self.__node_stack.pop()

    # Enter a parse tree produced by OpenSCENARIO2Parser#onMember.
    def enterOnMember(self, ctx: OpenSCENARIO2Parser.OnMemberContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#onMember.
    def exitOnMember(self, ctx: OpenSCENARIO2Parser.OnMemberContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#doDirective.
    def enterDoDirective(self, ctx: OpenSCENARIO2Parser.DoDirectiveContext):
        self.__node_stack.append(self.__cur_node)

        do_directive_scope = DoDirectiveScope(self.__current_scope)
        self.__current_scope.define(do_directive_scope, ctx.start)
        self.__current_scope = do_directive_scope

        node = ast_node.DoDirective()
        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(self.__current_scope)

        self.__cur_node.set_children(node)
        self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#doDirective.
    def exitDoDirective(self, ctx: OpenSCENARIO2Parser.DoDirectiveContext):
        self.__cur_node = self.__node_stack.pop()
        self.__current_scope = self.__current_scope.get_enclosing_scope()

    # Enter a parse tree produced by OpenSCENARIO2Parser#doMember.
    def enterDoMember(self, ctx: OpenSCENARIO2Parser.DoMemberContext):
        self.__node_stack.append(self.__cur_node)
        label_name = None

        if ctx.labelName():
            label_name = ctx.labelName().getText()

        composition_operator = None
        if ctx.composition():
            composition_operator = ctx.composition().compositionOperator().getText()

        if composition_operator is not None:
            domember = DoMemberSymbol(
                label_name, self.__current_scope, composition_operator
            )
            self.__current_scope.define(domember, ctx.start)
            self.__current_scope = domember

            node = ast_node.DoMember(label_name, composition_operator)
            node.set_loc(ctx.start.line, ctx.start.column)
            node.set_scope(self.__current_scope)

            self.__cur_node.set_children(node)
            self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#doMember.
    def exitDoMember(self, ctx: OpenSCENARIO2Parser.DoMemberContext):
        self.__cur_node = self.__node_stack.pop()
        if ctx.composition() is not None:
            self.__current_scope = self.__current_scope.get_enclosing_scope()

    # Enter a parse tree produced by OpenSCENARIO2Parser#composition.
    def enterComposition(self, ctx: OpenSCENARIO2Parser.CompositionContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#composition.
    def exitComposition(self, ctx: OpenSCENARIO2Parser.CompositionContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#compositionOperator.
    def enterCompositionOperator(
        self, ctx: OpenSCENARIO2Parser.CompositionOperatorContext
    ):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#compositionOperator.
    def exitCompositionOperator(
        self, ctx: OpenSCENARIO2Parser.CompositionOperatorContext
    ):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#behaviorInvocation.
    def enterBehaviorInvocation(
        self, ctx: OpenSCENARIO2Parser.BehaviorInvocationContext
    ):
        self.__node_stack.append(self.__cur_node)
        actor = None
        name = ""
        behavior_name = ctx.behaviorName().getText()
        if ctx.actorExpression():
            actor = ctx.actorExpression().getText()
            name += actor + "."

        name += behavior_name

        # Find the scope of type_name
        scope = self.__current_scope.resolve(name)

        if scope:
            pass
        else:
            msg = "behavior name: " + name + " is not defined!"
            pass

        node = ast_node.BehaviorInvocation(actor, behavior_name)
        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(scope)

        self.__cur_node.set_children(node)
        self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#behaviorInvocation.
    def exitBehaviorInvocation(
        self, ctx: OpenSCENARIO2Parser.BehaviorInvocationContext
    ):
        self.__cur_node = self.__node_stack.pop()

    # Enter a parse tree produced by OpenSCENARIO2Parser#behaviorWithDeclaration.
    def enterBehaviorWithDeclaration(
        self, ctx: OpenSCENARIO2Parser.BehaviorWithDeclarationContext
    ):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#behaviorWithDeclaration.
    def exitBehaviorWithDeclaration(
        self, ctx: OpenSCENARIO2Parser.BehaviorWithDeclarationContext
    ):
        # self.__current_scope = self.__current_scope.get_enclosing_scope()
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#behaviorWithMember.
    def enterBehaviorWithMember(
        self, ctx: OpenSCENARIO2Parser.BehaviorWithMemberContext
    ):
        self.__node_stack.append(self.__cur_node)

    # Exit a parse tree produced by OpenSCENARIO2Parser#behaviorWithMember.
    def exitBehaviorWithMember(
        self, ctx: OpenSCENARIO2Parser.BehaviorWithMemberContext
    ):
        self.__cur_node = self.__node_stack.pop()

    # Enter a parse tree produced by OpenSCENARIO2Parser#labelName.
    def enterLabelName(self, ctx: OpenSCENARIO2Parser.LabelNameContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#labelName.
    def exitLabelName(self, ctx: OpenSCENARIO2Parser.LabelNameContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#actorExpression.
    def enterActorExpression(self, ctx: OpenSCENARIO2Parser.ActorExpressionContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#actorExpression.
    def exitActorExpression(self, ctx: OpenSCENARIO2Parser.ActorExpressionContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#waitDirective.
    def enterWaitDirective(self, ctx: OpenSCENARIO2Parser.WaitDirectiveContext):
        self.__node_stack.append(self.__cur_node)

        wait_scope = WaitSymbol(self.__current_scope)
        self.__current_scope.define(wait_scope, ctx.start)
        self.__current_scope = wait_scope

        node = ast_node.WaitDirective()
        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(self.__current_scope)

        self.__cur_node.set_children(node)
        self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#waitDirective.
    def exitWaitDirective(self, ctx: OpenSCENARIO2Parser.WaitDirectiveContext):
        self.__cur_node = self.__node_stack.pop()
        self.__current_scope = self.__current_scope.get_enclosing_scope()

    # Enter a parse tree produced by OpenSCENARIO2Parser#emitDirective.
    def enterEmitDirective(self, ctx: OpenSCENARIO2Parser.EmitDirectiveContext):
        self.__node_stack.append(self.__cur_node)
        event_name = ctx.eventName().getText()
        node = ast_node.EmitDirective(event_name)
        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(self.__current_scope)

        self.__cur_node.set_children(node)
        self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#emitDirective.
    def exitEmitDirective(self, ctx: OpenSCENARIO2Parser.EmitDirectiveContext):
        self.__cur_node = self.__node_stack.pop()

    # Enter a parse tree produced by OpenSCENARIO2Parser#callDirective.
    def enterCallDirective(self, ctx: OpenSCENARIO2Parser.CallDirectiveContext):
        self.__node_stack.append(self.__cur_node)
        method_name = ctx.methodInvocation().postfixExp().getText()

        node = ast_node.CallDirective(method_name)
        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(self.__current_scope)

        self.__cur_node.set_children(node)
        self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#callDirective.
    def exitCallDirective(self, ctx: OpenSCENARIO2Parser.CallDirectiveContext):
        self.__cur_node = self.__node_stack.pop()

    # Enter a parse tree produced by OpenSCENARIO2Parser#untilDirective.
    def enterUntilDirective(self, ctx: OpenSCENARIO2Parser.UntilDirectiveContext):
        self.__node_stack.append(self.__cur_node)
        node = ast_node.UntilDirective()
        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(self.__current_scope)

        self.__cur_node.set_children(node)
        self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#untilDirective.
    def exitUntilDirective(self, ctx: OpenSCENARIO2Parser.UntilDirectiveContext):
        self.__cur_node = self.__node_stack.pop()

    # Enter a parse tree produced by OpenSCENARIO2Parser#methodInvocation.
    def enterMethodInvocation(self, ctx: OpenSCENARIO2Parser.MethodInvocationContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#methodInvocation.
    def exitMethodInvocation(self, ctx: OpenSCENARIO2Parser.MethodInvocationContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#methodDeclaration.
    def enterMethodDeclaration(self, ctx: OpenSCENARIO2Parser.MethodDeclarationContext):
        self.__node_stack.append(self.__cur_node)
        method_name = ctx.methodName().getText()
        return_type = None
        if ctx.returnType():
            return_type = ctx.returnType().getText()

        method = MethodSymbol(method_name, self.__current_scope)

        node = ast_node.MethodDeclaration(method_name, return_type)
        node.set_loc(ctx.start.line, ctx.start.column)

        method.declaration_address = node
        self.__current_scope.define(method, ctx.start)
        self.__current_scope = method
        node.set_scope(self.__current_scope)

        self.__cur_node.set_children(node)
        self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#methodDeclaration.
    def exitMethodDeclaration(self, ctx: OpenSCENARIO2Parser.MethodDeclarationContext):
        self.__cur_node = self.__node_stack.pop()
        self.__current_scope = self.__current_scope.get_enclosing_scope()

    # Enter a parse tree produced by OpenSCENARIO2Parser#returnType.
    def enterReturnType(self, ctx: OpenSCENARIO2Parser.ReturnTypeContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#returnType.
    def exitReturnType(self, ctx: OpenSCENARIO2Parser.ReturnTypeContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#methodImplementation.
    def enterMethodImplementation(
        self, ctx: OpenSCENARIO2Parser.MethodImplementationContext
    ):
        self.__node_stack.append(self.__cur_node)
        qualifier = None
        if ctx.methodQualifier():
            qualifier = ctx.methodQualifier().getText()

        if ctx.expression():
            _type = "expression"
        elif ctx.structuredIdentifier():
            _type = "external"
        else:
            _type = "undefined"

        external_name = None
        if ctx.structuredIdentifier():
            external_name = ctx.structuredIdentifier().getText()

        node = ast_node.MethodBody(qualifier, _type, external_name)
        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(self.__current_scope)

        self.__cur_node.set_children(node)
        self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#methodImplementation.
    def exitMethodImplementation(
        self, ctx: OpenSCENARIO2Parser.MethodImplementationContext
    ):
        self.__cur_node = self.__node_stack.pop()

    # Enter a parse tree produced by OpenSCENARIO2Parser#methodQualifier.
    def enterMethodQualifier(self, ctx: OpenSCENARIO2Parser.MethodQualifierContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#methodQualifier.
    def exitMethodQualifier(self, ctx: OpenSCENARIO2Parser.MethodQualifierContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#methodName.
    def enterMethodName(self, ctx: OpenSCENARIO2Parser.MethodNameContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#methodName.
    def exitMethodName(self, ctx: OpenSCENARIO2Parser.MethodNameContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#coverageDeclaration.
    def enterCoverageDeclaration(
        self, ctx: OpenSCENARIO2Parser.CoverageDeclarationContext
    ):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#coverageDeclaration.
    def exitCoverageDeclaration(
        self, ctx: OpenSCENARIO2Parser.CoverageDeclarationContext
    ):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#coverDeclaration.
    def enterCoverDeclaration(self, ctx: OpenSCENARIO2Parser.CoverDeclarationContext):
        self.__node_stack.append(self.__cur_node)
        target_name = None
        if ctx.targetName():
            target_name = ctx.targetName().getText()

        node = ast_node.coverDeclaration(target_name)
        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(self.__current_scope)

        self.__cur_node.set_children(node)
        self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#coverDeclaration.
    def exitCoverDeclaration(self, ctx: OpenSCENARIO2Parser.CoverDeclarationContext):
        self.__cur_node = self.__node_stack.pop()

    # Enter a parse tree produced by OpenSCENARIO2Parser#recordDeclaration.
    def enterRecordDeclaration(self, ctx: OpenSCENARIO2Parser.RecordDeclarationContext):
        self.__node_stack.append(self.__cur_node)
        target_name = None
        if ctx.targetName():
            target_name = ctx.targetName().getText()

        node = ast_node.recordDeclaration(target_name)
        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(self.__current_scope)

        self.__cur_node.set_children(node)
        self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#recordDeclaration.
    def exitRecordDeclaration(self, ctx: OpenSCENARIO2Parser.RecordDeclarationContext):
        self.__cur_node = self.__node_stack.pop()

    # Enter a parse tree produced by OpenSCENARIO2Parser#coverageExpression.
    def enterCoverageExpression(
        self, ctx: OpenSCENARIO2Parser.CoverageExpressionContext
    ):
        self.__node_stack.append(self.__cur_node)
        argument_name = "expression"
        node = ast_node.NamedArgument(argument_name)
        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(self.__current_scope)

        self.__cur_node.set_children(node)
        self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#coverageExpression.
    def exitCoverageExpression(
        self, ctx: OpenSCENARIO2Parser.CoverageExpressionContext
    ):
        self.__cur_node = self.__node_stack.pop()

    # Enter a parse tree produced by OpenSCENARIO2Parser#coverageUnit.
    def enterCoverageUnit(self, ctx: OpenSCENARIO2Parser.CoverageUnitContext):
        self.__node_stack.append(self.__cur_node)
        argument_name = "unit"
        node = ast_node.NamedArgument(argument_name)
        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(self.__current_scope)

        unit_name = ast_node.Identifier(ctx.Identifier().getText())
        unit_name.set_loc(ctx.start.line, ctx.start.column)
        unit_name.set_scope(self.__current_scope)
        node.set_children(unit_name)

        self.__cur_node.set_children(node)
        self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#coverageUnit.
    def exitCoverageUnit(self, ctx: OpenSCENARIO2Parser.CoverageUnitContext):
        self.__cur_node = self.__node_stack.pop()

    # Enter a parse tree produced by OpenSCENARIO2Parser#coverageRange.
    def enterCoverageRange(self, ctx: OpenSCENARIO2Parser.CoverageRangeContext):
        self.__node_stack.append(self.__cur_node)
        argument_name = "range"
        node = ast_node.NamedArgument(argument_name)
        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(self.__current_scope)

        self.__cur_node.set_children(node)
        self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#coverageRange.
    def exitCoverageRange(self, ctx: OpenSCENARIO2Parser.CoverageRangeContext):
        self.__cur_node = self.__node_stack.pop()

    # Enter a parse tree produced by OpenSCENARIO2Parser#coverageEvery.
    def enterCoverageEvery(self, ctx: OpenSCENARIO2Parser.CoverageEveryContext):
        self.__node_stack.append(self.__cur_node)
        argument_name = "every"
        node = ast_node.NamedArgument(argument_name)
        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(self.__current_scope)

        self.__cur_node.set_children(node)
        self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#coverageEvery.
    def exitCoverageEvery(self, ctx: OpenSCENARIO2Parser.CoverageEveryContext):
        self.__cur_node = self.__node_stack.pop()

    # Enter a parse tree produced by OpenSCENARIO2Parser#coverageEvent.
    def enterCoverageEvent(self, ctx: OpenSCENARIO2Parser.CoverageEventContext):
        self.__node_stack.append(self.__cur_node)
        argument_name = "event"
        node = ast_node.NamedArgument(argument_name)
        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(self.__current_scope)

        self.__cur_node.set_children(node)
        self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#coverageEvent.
    def exitCoverageEvent(self, ctx: OpenSCENARIO2Parser.CoverageEventContext):
        self.__cur_node = self.__node_stack.pop()

    # Enter a parse tree produced by OpenSCENARIO2Parser#coverageNameArgument.
    def enterCoverageNameArgument(
        self, ctx: OpenSCENARIO2Parser.CoverageNameArgumentContext
    ):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#coverageNameArgument.
    def exitCoverageNameArgument(
        self, ctx: OpenSCENARIO2Parser.CoverageNameArgumentContext
    ):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#targetName.
    def enterTargetName(self, ctx: OpenSCENARIO2Parser.TargetNameContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#targetName.
    def exitTargetName(self, ctx: OpenSCENARIO2Parser.TargetNameContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#expression.
    def enterExpression(self, ctx: OpenSCENARIO2Parser.ExpressionContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#expression.
    def exitExpression(self, ctx: OpenSCENARIO2Parser.ExpressionContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#ternaryOpExp.
    def enterTernaryOpExp(self, ctx: OpenSCENARIO2Parser.TernaryOpExpContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#ternaryOpExp.
    def exitTernaryOpExp(self, ctx: OpenSCENARIO2Parser.TernaryOpExpContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#implication.
    def enterImplication(self, ctx: OpenSCENARIO2Parser.ImplicationContext):
        self.__node_stack.append(self.__cur_node)
        if len(ctx.disjunction()) > 1:
            operator = "=>"
            node = ast_node.LogicalExpression(operator)
            node.set_loc(ctx.start.line, ctx.start.column)
            node.set_scope(self.__current_scope)

            self.__cur_node.set_children(node)
            self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#implication.
    def exitImplication(self, ctx: OpenSCENARIO2Parser.ImplicationContext):
        self.__cur_node = self.__node_stack.pop()

    # Enter a parse tree produced by OpenSCENARIO2Parser#disjunction.
    def enterDisjunction(self, ctx: OpenSCENARIO2Parser.DisjunctionContext):
        self.__node_stack.append(self.__cur_node)
        if len(ctx.conjunction()) > 1:
            operator = "or"
            node = ast_node.LogicalExpression(operator)
            node.set_loc(ctx.start.line, ctx.start.column)
            node.set_scope(self.__current_scope)

            self.__cur_node.set_children(node)
            self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#disjunction.
    def exitDisjunction(self, ctx: OpenSCENARIO2Parser.DisjunctionContext):
        self.__cur_node = self.__node_stack.pop()

    # Enter a parse tree produced by OpenSCENARIO2Parser#conjunction.
    def enterConjunction(self, ctx: OpenSCENARIO2Parser.ConjunctionContext):
        self.__node_stack.append(self.__cur_node)
        if len(ctx.inversion()) > 1:
            operator = "and"
            node = ast_node.LogicalExpression(operator)
            node.set_loc(ctx.start.line, ctx.start.column)
            node.set_scope(self.__current_scope)

            self.__cur_node.set_children(node)
            self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#conjunction.
    def exitConjunction(self, ctx: OpenSCENARIO2Parser.ConjunctionContext):
        self.__cur_node = self.__node_stack.pop()

    # Enter a parse tree produced by OpenSCENARIO2Parser#inversion.
    def enterInversion(self, ctx: OpenSCENARIO2Parser.InversionContext):
        self.__node_stack.append(self.__cur_node)
        if ctx.relation() == None:
            operator = "not"
            node = ast_node.LogicalExpression(operator)
            node.set_loc(ctx.start.line, ctx.start.column)
            node.set_scope(self.__current_scope)

            self.__cur_node.set_children(node)
            self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#inversion.
    def exitInversion(self, ctx: OpenSCENARIO2Parser.InversionContext):
        self.__cur_node = self.__node_stack.pop()

    # Enter a parse tree produced by OpenSCENARIO2Parser#relation.
    def enterRelation(self, ctx: OpenSCENARIO2Parser.RelationContext):
        self.__node_stack.append(self.__cur_node)

    # Exit a parse tree produced by OpenSCENARIO2Parser#relation.
    def exitRelation(self, ctx: OpenSCENARIO2Parser.RelationContext):
        self.__cur_node = self.__node_stack.pop()

    # Enter a parse tree produced by OpenSCENARIO2Parser#relationExp.
    def enterRelationExp(self, ctx: OpenSCENARIO2Parser.RelationExpContext):
        self.__node_stack.append(self.__cur_node)
        operator = ctx.relationalOp().getText()
        node = ast_node.RelationExpression(operator)
        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(self.__current_scope)

        self.__cur_node.set_children(node)
        self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#relationExp.
    def exitRelationExp(self, ctx: OpenSCENARIO2Parser.RelationExpContext):
        self.__cur_node = self.__node_stack.pop()

    # Enter a parse tree produced by OpenSCENARIO2Parser#relationalOp.
    def enterRelationalOp(self, ctx: OpenSCENARIO2Parser.RelationalOpContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#relationalOp.
    def exitRelationalOp(self, ctx: OpenSCENARIO2Parser.RelationalOpContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#sum.
    def enterSum(self, ctx: OpenSCENARIO2Parser.SumContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#sum.
    def exitSum(self, ctx: OpenSCENARIO2Parser.SumContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#additiveExp.
    def enterAdditiveExp(self, ctx: OpenSCENARIO2Parser.AdditiveExpContext):
        self.__node_stack.append(self.__cur_node)
        operator = ctx.additiveOp().getText()
        node = ast_node.BinaryExpression(operator)
        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(self.__current_scope)

        self.__cur_node.set_children(node)
        self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#additiveExp.
    def exitAdditiveExp(self, ctx: OpenSCENARIO2Parser.AdditiveExpContext):
        self.__cur_node = self.__node_stack.pop()

    # Enter a parse tree produced by OpenSCENARIO2Parser#additiveOp.
    def enterAdditiveOp(self, ctx: OpenSCENARIO2Parser.AdditiveOpContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#additiveOp.
    def exitAdditiveOp(self, ctx: OpenSCENARIO2Parser.AdditiveOpContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#multiplicativeExp.
    def enterMultiplicativeExp(self, ctx: OpenSCENARIO2Parser.MultiplicativeExpContext):
        self.__node_stack.append(self.__cur_node)
        operator = ctx.multiplicativeOp().getText()
        node = ast_node.BinaryExpression(operator)
        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(self.__current_scope)

        self.__cur_node.set_children(node)
        self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#multiplicativeExp.
    def exitMultiplicativeExp(self, ctx: OpenSCENARIO2Parser.MultiplicativeExpContext):
        self.__cur_node = self.__node_stack.pop()

    # Enter a parse tree produced by OpenSCENARIO2Parser#term.
    def enterTerm(self, ctx: OpenSCENARIO2Parser.TermContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#term.
    def exitTerm(self, ctx: OpenSCENARIO2Parser.TermContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#multiplicativeOp.
    def enterMultiplicativeOp(self, ctx: OpenSCENARIO2Parser.MultiplicativeOpContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#multiplicativeOp.
    def exitMultiplicativeOp(self, ctx: OpenSCENARIO2Parser.MultiplicativeOpContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#factor.
    def enterFactor(self, ctx: OpenSCENARIO2Parser.FactorContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#factor.
    def exitFactor(self, ctx: OpenSCENARIO2Parser.FactorContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#primaryExpression.
    def enterPrimaryExpression(self, ctx: OpenSCENARIO2Parser.PrimaryExpressionContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#primaryExpression.
    def exitPrimaryExpression(self, ctx: OpenSCENARIO2Parser.PrimaryExpressionContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#castExpression.
    def enterCastExpression(self, ctx: OpenSCENARIO2Parser.CastExpressionContext):
        self.__node_stack.append(self.__cur_node)
        object = ctx.postfixExp().getText()
        target_type = ctx.typeDeclarator().getText()
        node = ast_node.CastExpression(object, target_type)
        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(self.__current_scope)

        self.__cur_node.set_children(node)
        self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#castExpression.
    def exitCastExpression(self, ctx: OpenSCENARIO2Parser.CastExpressionContext):
        self.__cur_node = self.__node_stack.pop()

    # Enter a parse tree produced by OpenSCENARIO2Parser#functionApplicationExpression.
    def enterFunctionApplicationExpression(
        self, ctx: OpenSCENARIO2Parser.FunctionApplicationExpressionContext
    ):
        self.__node_stack.append(self.__cur_node)
        func_name = ctx.postfixExp().getText()
        scope = self.__current_scope
        func_name_list = func_name.split(".", 1)
        if len(func_name_list) > 1:
            scope = scope.resolve(func_name_list[0])
        node = ast_node.FunctionApplicationExpression(func_name)
        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(scope)

        self.__cur_node.set_children(node)
        self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#functionApplicationExpression.
    def exitFunctionApplicationExpression(
        self, ctx: OpenSCENARIO2Parser.FunctionApplicationExpressionContext
    ):
        self.__cur_node = self.__node_stack.pop()

    # Enter a parse tree produced by OpenSCENARIO2Parser#fieldAccessExpression.
    def enterFieldAccessExpression(
        self, ctx: OpenSCENARIO2Parser.FieldAccessExpressionContext
    ):
        self.__node_stack.append(self.__cur_node)
        field_name = ctx.postfixExp().getText() + "." + ctx.fieldName().getText()

        if ctx.postfixExp().getText() == "it":
            field_name = self.__current_scope.get_enclosing_scope().type
            scope = None
            if self.__current_scope.resolve(field_name):
                scope = self.__current_scope.resolve(field_name)
                if ctx.fieldName().getText() in scope.symbols:
                    pass
                else:
                    msg = (
                        ctx.fieldName().getText()
                        + " is not defined in scope: "
                        + self.__current_scope.get_enclosing_scope().type
                    )
                    LOG_ERROR(msg, ctx.start)
            else:
                msg = "it -> " + field_name + " is not defined!"
                LOG_ERROR(msg, ctx.start)

        node = ast_node.FieldAccessExpression(field_name)
        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(self.__current_scope)

        self.__cur_node.set_children(node)
        self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#fieldAccessExpression.
    def exitFieldAccessExpression(
        self, ctx: OpenSCENARIO2Parser.FieldAccessExpressionContext
    ):
        self.__cur_node = self.__node_stack.pop()

    # Enter a parse tree produced by OpenSCENARIO2Parser#elementAccessExpression.
    def enterElementAccessExpression(
        self, ctx: OpenSCENARIO2Parser.ElementAccessExpressionContext
    ):
        self.__node_stack.append(self.__cur_node)
        list_name = ctx.postfixExp().getText()
        index = ctx.expression().getText()
        node = ast_node.ElementAccessExpression(list_name, index)
        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(self.__current_scope)

        self.__cur_node.set_children(node)
        self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#elementAccessExpression.
    def exitElementAccessExpression(
        self, ctx: OpenSCENARIO2Parser.ElementAccessExpressionContext
    ):
        self.__cur_node = self.__node_stack.pop()

    # Enter a parse tree produced by OpenSCENARIO2Parser#typeTestExpression.
    def enterTypeTestExpression(
        self, ctx: OpenSCENARIO2Parser.TypeTestExpressionContext
    ):
        self.__node_stack.append(self.__cur_node)
        object = ctx.postfixExp().getText()
        target_type = ctx.typeDeclarator().getText()
        node = ast_node.TypeTestExpression(object, target_type)
        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(self.__current_scope)

        self.__cur_node.set_children(node)
        self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#typeTestExpression.
    def exitTypeTestExpression(
        self, ctx: OpenSCENARIO2Parser.TypeTestExpressionContext
    ):
        self.__cur_node = self.__node_stack.pop()

    # Enter a parse tree produced by OpenSCENARIO2Parser#fieldAccess.
    def enterFieldAccess(self, ctx: OpenSCENARIO2Parser.FieldAccessContext):
        self.__node_stack.append(self.__cur_node)
        field_name = ctx.postfixExp().getText() + "." + ctx.fieldName().getText()
        node = ast_node.FieldAccessExpression(field_name)
        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(self.__current_scope)

        self.__cur_node.set_children(node)
        self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#fieldAccess.
    def exitFieldAccess(self, ctx: OpenSCENARIO2Parser.FieldAccessContext):
        self.__cur_node = self.__node_stack.pop()

    # Enter a parse tree produced by OpenSCENARIO2Parser#primaryExp.
    def enterPrimaryExp(self, ctx: OpenSCENARIO2Parser.PrimaryExpContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#primaryExp.
    def exitPrimaryExp(self, ctx: OpenSCENARIO2Parser.PrimaryExpContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#valueExp.
    def enterValueExp(self, ctx: OpenSCENARIO2Parser.ValueExpContext):
        self.__node_stack.append(self.__cur_node)
        value = None
        node = None
        if ctx.FloatLiteral():
            value = ctx.FloatLiteral().getText()
            node = ast_node.FloatLiteral(value)
        elif ctx.BoolLiteral():
            value = ctx.BoolLiteral().getText()
            node = ast_node.BoolLiteral(value)
        elif ctx.StringLiteral():
            value = ctx.StringLiteral().getText()
            value = value.strip('"')
            node = ast_node.StringLiteral(value)

        if node is not None:
            node.set_loc(ctx.start.line, ctx.start.column)
            node.set_scope(self.__current_scope)

            self.__cur_node.set_children(node)
            self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#valueExp.
    def exitValueExp(self, ctx: OpenSCENARIO2Parser.ValueExpContext):
        self.__cur_node = self.__node_stack.pop()

    # Enter a parse tree produced by OpenSCENARIO2Parser#listConstructor.
    def enterListConstructor(self, ctx: OpenSCENARIO2Parser.ListConstructorContext):
        self.__node_stack.append(self.__cur_node)
        node = ast_node.ListExpression()
        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(self.__current_scope)

        self.__cur_node.set_children(node)
        self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#listConstructor.
    def exitListConstructor(self, ctx: OpenSCENARIO2Parser.ListConstructorContext):
        self.__cur_node = self.__node_stack.pop()

    # Enter a parse tree produced by OpenSCENARIO2Parser#rangeConstructor.
    def enterRangeConstructor(self, ctx: OpenSCENARIO2Parser.RangeConstructorContext):
        self.__node_stack.append(self.__cur_node)
        node = ast_node.RangeExpression()
        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(self.__current_scope)

        self.__cur_node.set_children(node)
        self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#rangeConstructor.
    def exitRangeConstructor(self, ctx: OpenSCENARIO2Parser.RangeConstructorContext):
        self.__cur_node = self.__node_stack.pop()

    # Enter a parse tree produced by OpenSCENARIO2Parser#identifierReference.
    def enterIdentifierReference(
        self, ctx: OpenSCENARIO2Parser.IdentifierReferenceContext
    ):
        self.__node_stack.append(self.__cur_node)

        field_name = []
        scope = None
        for fn in ctx.fieldName():
            name = fn.Identifier().getText()
            if scope is None:
                if self.__current_scope.resolve(name):
                    scope = self.__current_scope.resolve(name)
                else:
                    msg = name + " is not defined!"
                    LOG_ERROR(msg, ctx.start)
            else:
                if issubclass(type(scope), TypedSymbol):
                    if self.__current_scope.resolve(scope.type):
                        scope = self.__current_scope.resolve(scope.type)
                    if name in scope.symbols:
                        if scope.symbols[name].value:
                            scope = scope.symbols[name].value
                        else:
                            msg = name + ": value is None!"
                            LOG_ERROR(msg, ctx.start)
                    else:
                        msg = name + " is not defined!"
                        LOG_ERROR(msg, ctx.start)
                else:
                    scope = self.__current_scope.resolve(scope)

            field_name.append(name)

        id_name = ".".join(field_name)

        node = ast_node.IdentifierReference(id_name)
        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(scope)

        self.__cur_node.set_children(node)
        self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#identifierReference.
    def exitIdentifierReference(
        self, ctx: OpenSCENARIO2Parser.IdentifierReferenceContext
    ):
        self.__cur_node = self.__node_stack.pop()

    # Enter a parse tree produced by OpenSCENARIO2Parser#argumentListSpecification.
    def enterArgumentListSpecification(
        self, ctx: OpenSCENARIO2Parser.ArgumentListSpecificationContext
    ):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#argumentListSpecification.
    def exitArgumentListSpecification(
        self, ctx: OpenSCENARIO2Parser.ArgumentListSpecificationContext
    ):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#argumentSpecification.
    def enterArgumentSpecification(
        self, ctx: OpenSCENARIO2Parser.ArgumentSpecificationContext
    ):
        self.__node_stack.append(self.__cur_node)
        argument_name = ctx.argumentName().getText()
        argument_type = ctx.typeDeclarator().getText()
        default_value = None
        if ctx.defaultValue():
            default_value = ctx.defaultValue().getText()

        scope = self.__current_scope.resolve(argument_type)
        if scope:
            pass
        elif (
            argument_type == "int"
            or argument_type == "uint"
            or argument_type == "float"
            or argument_type == "bool"
            or argument_type == "string"
        ):
            pass
        else:
            msg = "Argument Type " + argument_type + " is not defined!"
            LOG_ERROR(msg, ctx.start)

        argument = ArgumentSpecificationSymbol(
            argument_name, self.__current_scope, argument_type, default_value
        )
        self.__current_scope.define(argument, ctx.start)
        self.__current_scope = argument

        node = ast_node.Argument(argument_name, argument_type, default_value)
        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(self.__current_scope)

        self.__cur_node.set_children(node)
        self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#argumentSpecification.
    def exitArgumentSpecification(
        self, ctx: OpenSCENARIO2Parser.ArgumentSpecificationContext
    ):
        self.__cur_node = self.__node_stack.pop()
        self.__current_scope = self.__current_scope.get_enclosing_scope()

    # Enter a parse tree produced by OpenSCENARIO2Parser#argumentName.
    def enterArgumentName(self, ctx: OpenSCENARIO2Parser.ArgumentNameContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#argumentName.
    def exitArgumentName(self, ctx: OpenSCENARIO2Parser.ArgumentNameContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#argumentList.
    def enterArgumentList(self, ctx: OpenSCENARIO2Parser.ArgumentListContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#argumentList.
    def exitArgumentList(self, ctx: OpenSCENARIO2Parser.ArgumentListContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#positionalArgument.
    def enterPositionalArgument(
        self, ctx: OpenSCENARIO2Parser.PositionalArgumentContext
    ):
        self.__node_stack.append(self.__cur_node)

        node = ast_node.PositionalArgument()
        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(self.__current_scope)

        self.__cur_node.set_children(node)
        self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#positionalArgument.
    def exitPositionalArgument(
        self, ctx: OpenSCENARIO2Parser.PositionalArgumentContext
    ):
        self.__cur_node = self.__node_stack.pop()

    # Enter a parse tree produced by OpenSCENARIO2Parser#namedArgument.
    def enterNamedArgument(self, ctx: OpenSCENARIO2Parser.NamedArgumentContext):
        self.__node_stack.append(self.__cur_node)
        argument_name = ctx.argumentName().getText()

        node = ast_node.NamedArgument(argument_name)
        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(self.__current_scope)

        self.__cur_node.set_children(node)
        self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#namedArgument.
    def exitNamedArgument(self, ctx: OpenSCENARIO2Parser.NamedArgumentContext):
        self.__cur_node = self.__node_stack.pop()

    # Enter a parse tree produced by OpenSCENARIO2Parser#physicalLiteral.
    def enterPhysicalLiteral(self, ctx: OpenSCENARIO2Parser.PhysicalLiteralContext):
        self.__node_stack.append(self.__cur_node)
        unit_name = ctx.Identifier().getText()
        value = None
        if ctx.FloatLiteral():
            value = ctx.FloatLiteral().getText()
        else:
            value = ctx.integerLiteral().getText()

        scope = self.__current_scope.resolve(unit_name)
        if scope and isinstance(scope, UnitSymbol):
            pass
        else:
            msg = "Unit " + unit_name + " is not defined!"
            LOG_WARNING(msg, ctx.start)

        node = ast_node.PhysicalLiteral(unit_name, value)
        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(self.__current_scope)

        self.__cur_node.set_children(node)
        self.__cur_node = node

        if ctx.FloatLiteral():
            self.__node_stack.append(self.__cur_node)
            float_value = ctx.FloatLiteral().getText()
            value_node = ast_node.FloatLiteral(float_value)
            value_node.set_loc(ctx.start.line, ctx.start.column)
            value_node.set_scope(self.__current_scope)
            node.set_children(value_node)
            self.__cur_node = self.__node_stack.pop()

    # Exit a parse tree produced by OpenSCENARIO2Parser#physicalLiteral.
    def exitPhysicalLiteral(self, ctx: OpenSCENARIO2Parser.PhysicalLiteralContext):
        self.__cur_node = self.__node_stack.pop()

    # Enter a parse tree produced by OpenSCENARIO2Parser#integerLiteral.
    def enterIntegerLiteral(self, ctx: OpenSCENARIO2Parser.IntegerLiteralContext):
        self.__node_stack.append(self.__cur_node)
        value = None
        type = "uint"
        if ctx.UintLiteral():
            value = ctx.UintLiteral().getText()
            type = "uint"
        elif ctx.HexUintLiteral():
            value = ctx.HexUintLiteral().getText()
            type = "hex"
        elif ctx.IntLiteral():
            value = ctx.IntLiteral().getText()
            type = "int"
        else:  # only the above three types of integer literal
            pass

        node = ast_node.IntegerLiteral(type, value)
        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(self.__current_scope)

        self.__cur_node.set_children(node)
        self.__cur_node = node

    # Exit a parse tree produced by OpenSCENARIO2Parser#integerLiteral.
    def exitIntegerLiteral(self, ctx: OpenSCENARIO2Parser.IntegerLiteralContext):
        self.__cur_node = self.__node_stack.pop()


del OpenSCENARIO2Parser
