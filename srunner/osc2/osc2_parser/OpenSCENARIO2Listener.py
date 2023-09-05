# Generated from .\OpenSCENARIO2.g4 by ANTLR 4.10.1
from antlr4 import *

if __name__ is not None and "." in __name__:
    from .OpenSCENARIO2Parser import OpenSCENARIO2Parser
else:
    from OpenSCENARIO2Parser import OpenSCENARIO2Parser


# This class defines a complete listener for a parse tree produced by OpenSCENARIO2Parser.
class OpenSCENARIO2Listener(ParseTreeListener):
    # Enter a parse tree produced by OpenSCENARIO2Parser#osc_file.
    def enterOsc_file(self, ctx: OpenSCENARIO2Parser.Osc_fileContext):
        pass

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
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#physicalTypeDeclaration.
    def exitPhysicalTypeDeclaration(
        self, ctx: OpenSCENARIO2Parser.PhysicalTypeDeclarationContext
    ):
        pass

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
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#sIBaseUnitSpecifier.
    def exitSIBaseUnitSpecifier(
        self, ctx: OpenSCENARIO2Parser.SIBaseUnitSpecifierContext
    ):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#unitDeclaration.
    def enterUnitDeclaration(self, ctx: OpenSCENARIO2Parser.UnitDeclarationContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#unitDeclaration.
    def exitUnitDeclaration(self, ctx: OpenSCENARIO2Parser.UnitDeclarationContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#unitSpecifier.
    def enterUnitSpecifier(self, ctx: OpenSCENARIO2Parser.UnitSpecifierContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#unitSpecifier.
    def exitUnitSpecifier(self, ctx: OpenSCENARIO2Parser.UnitSpecifierContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#sIUnitSpecifier.
    def enterSIUnitSpecifier(self, ctx: OpenSCENARIO2Parser.SIUnitSpecifierContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#sIUnitSpecifier.
    def exitSIUnitSpecifier(self, ctx: OpenSCENARIO2Parser.SIUnitSpecifierContext):
        pass

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
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#sIBaseExponent.
    def exitSIBaseExponent(self, ctx: OpenSCENARIO2Parser.SIBaseExponentContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#sIFactor.
    def enterSIFactor(self, ctx: OpenSCENARIO2Parser.SIFactorContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#sIFactor.
    def exitSIFactor(self, ctx: OpenSCENARIO2Parser.SIFactorContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#sIOffset.
    def enterSIOffset(self, ctx: OpenSCENARIO2Parser.SIOffsetContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#sIOffset.
    def exitSIOffset(self, ctx: OpenSCENARIO2Parser.SIOffsetContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#enumDeclaration.
    def enterEnumDeclaration(self, ctx: OpenSCENARIO2Parser.EnumDeclarationContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#enumDeclaration.
    def exitEnumDeclaration(self, ctx: OpenSCENARIO2Parser.EnumDeclarationContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#enumMemberDecl.
    def enterEnumMemberDecl(self, ctx: OpenSCENARIO2Parser.EnumMemberDeclContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#enumMemberDecl.
    def exitEnumMemberDecl(self, ctx: OpenSCENARIO2Parser.EnumMemberDeclContext):
        pass

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
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#enumValueReference.
    def exitEnumValueReference(
        self, ctx: OpenSCENARIO2Parser.EnumValueReferenceContext
    ):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#inheritsCondition.
    def enterInheritsCondition(self, ctx: OpenSCENARIO2Parser.InheritsConditionContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#inheritsCondition.
    def exitInheritsCondition(self, ctx: OpenSCENARIO2Parser.InheritsConditionContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#structDeclaration.
    def enterStructDeclaration(self, ctx: OpenSCENARIO2Parser.StructDeclarationContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#structDeclaration.
    def exitStructDeclaration(self, ctx: OpenSCENARIO2Parser.StructDeclarationContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#structInherts.
    def enterStructInherts(self, ctx: OpenSCENARIO2Parser.StructInhertsContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#structInherts.
    def exitStructInherts(self, ctx: OpenSCENARIO2Parser.StructInhertsContext):
        pass

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
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#actorDeclaration.
    def exitActorDeclaration(self, ctx: OpenSCENARIO2Parser.ActorDeclarationContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#actorInherts.
    def enterActorInherts(self, ctx: OpenSCENARIO2Parser.ActorInhertsContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#actorInherts.
    def exitActorInherts(self, ctx: OpenSCENARIO2Parser.ActorInhertsContext):
        pass

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
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#scenarioDeclaration.
    def exitScenarioDeclaration(
        self, ctx: OpenSCENARIO2Parser.ScenarioDeclarationContext
    ):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#scenarioInherts.
    def enterScenarioInherts(self, ctx: OpenSCENARIO2Parser.ScenarioInhertsContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#scenarioInherts.
    def exitScenarioInherts(self, ctx: OpenSCENARIO2Parser.ScenarioInhertsContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#scenarioMemberDecl.
    def enterScenarioMemberDecl(
        self, ctx: OpenSCENARIO2Parser.ScenarioMemberDeclContext
    ):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#scenarioMemberDecl.
    def exitScenarioMemberDecl(
        self, ctx: OpenSCENARIO2Parser.ScenarioMemberDeclContext
    ):
        pass

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
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#actionDeclaration.
    def exitActionDeclaration(self, ctx: OpenSCENARIO2Parser.ActionDeclarationContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#actionInherts.
    def enterActionInherts(self, ctx: OpenSCENARIO2Parser.ActionInhertsContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#actionInherts.
    def exitActionInherts(self, ctx: OpenSCENARIO2Parser.ActionInhertsContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#modifierDeclaration.
    def enterModifierDeclaration(
        self, ctx: OpenSCENARIO2Parser.ModifierDeclarationContext
    ):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#modifierDeclaration.
    def exitModifierDeclaration(
        self, ctx: OpenSCENARIO2Parser.ModifierDeclarationContext
    ):
        pass

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
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#enumTypeExtension.
    def exitEnumTypeExtension(self, ctx: OpenSCENARIO2Parser.EnumTypeExtensionContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#structuredTypeExtension.
    def enterStructuredTypeExtension(
        self, ctx: OpenSCENARIO2Parser.StructuredTypeExtensionContext
    ):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#structuredTypeExtension.
    def exitStructuredTypeExtension(
        self, ctx: OpenSCENARIO2Parser.StructuredTypeExtensionContext
    ):
        pass

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
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#globalParameterDeclaration.
    def exitGlobalParameterDeclaration(
        self, ctx: OpenSCENARIO2Parser.GlobalParameterDeclarationContext
    ):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#typeDeclarator.
    def enterTypeDeclarator(self, ctx: OpenSCENARIO2Parser.TypeDeclaratorContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#typeDeclarator.
    def exitTypeDeclarator(self, ctx: OpenSCENARIO2Parser.TypeDeclaratorContext):
        pass

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
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#typeName.
    def exitTypeName(self, ctx: OpenSCENARIO2Parser.TypeNameContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#eventDeclaration.
    def enterEventDeclaration(self, ctx: OpenSCENARIO2Parser.EventDeclarationContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#eventDeclaration.
    def exitEventDeclaration(self, ctx: OpenSCENARIO2Parser.EventDeclarationContext):
        pass

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
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#eventReference.
    def exitEventReference(self, ctx: OpenSCENARIO2Parser.EventReferenceContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#eventFieldDecl.
    def enterEventFieldDecl(self, ctx: OpenSCENARIO2Parser.EventFieldDeclContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#eventFieldDecl.
    def exitEventFieldDecl(self, ctx: OpenSCENARIO2Parser.EventFieldDeclContext):
        pass

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
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#eventCondition.
    def exitEventCondition(self, ctx: OpenSCENARIO2Parser.EventConditionContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#riseExpression.
    def enterRiseExpression(self, ctx: OpenSCENARIO2Parser.RiseExpressionContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#riseExpression.
    def exitRiseExpression(self, ctx: OpenSCENARIO2Parser.RiseExpressionContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#fallExpression.
    def enterFallExpression(self, ctx: OpenSCENARIO2Parser.FallExpressionContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#fallExpression.
    def exitFallExpression(self, ctx: OpenSCENARIO2Parser.FallExpressionContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#elapsedExpression.
    def enterElapsedExpression(self, ctx: OpenSCENARIO2Parser.ElapsedExpressionContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#elapsedExpression.
    def exitElapsedExpression(self, ctx: OpenSCENARIO2Parser.ElapsedExpressionContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#everyExpression.
    def enterEveryExpression(self, ctx: OpenSCENARIO2Parser.EveryExpressionContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#everyExpression.
    def exitEveryExpression(self, ctx: OpenSCENARIO2Parser.EveryExpressionContext):
        pass

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
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#parameterDeclaration.
    def exitParameterDeclaration(
        self, ctx: OpenSCENARIO2Parser.ParameterDeclarationContext
    ):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#variableDeclaration.
    def enterVariableDeclaration(
        self, ctx: OpenSCENARIO2Parser.VariableDeclarationContext
    ):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#variableDeclaration.
    def exitVariableDeclaration(
        self, ctx: OpenSCENARIO2Parser.VariableDeclarationContext
    ):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#sampleExpression.
    def enterSampleExpression(self, ctx: OpenSCENARIO2Parser.SampleExpressionContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#sampleExpression.
    def exitSampleExpression(self, ctx: OpenSCENARIO2Parser.SampleExpressionContext):
        pass

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
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#keepConstraintDeclaration.
    def exitKeepConstraintDeclaration(
        self, ctx: OpenSCENARIO2Parser.KeepConstraintDeclarationContext
    ):
        pass

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
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#removeDefaultDeclaration.
    def exitRemoveDefaultDeclaration(
        self, ctx: OpenSCENARIO2Parser.RemoveDefaultDeclarationContext
    ):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#parameterReference.
    def enterParameterReference(
        self, ctx: OpenSCENARIO2Parser.ParameterReferenceContext
    ):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#parameterReference.
    def exitParameterReference(
        self, ctx: OpenSCENARIO2Parser.ParameterReferenceContext
    ):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#modifierInvocation.
    def enterModifierInvocation(
        self, ctx: OpenSCENARIO2Parser.ModifierInvocationContext
    ):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#modifierInvocation.
    def exitModifierInvocation(
        self, ctx: OpenSCENARIO2Parser.ModifierInvocationContext
    ):
        pass

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
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#onDirective.
    def exitOnDirective(self, ctx: OpenSCENARIO2Parser.OnDirectiveContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#onMember.
    def enterOnMember(self, ctx: OpenSCENARIO2Parser.OnMemberContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#onMember.
    def exitOnMember(self, ctx: OpenSCENARIO2Parser.OnMemberContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#doDirective.
    def enterDoDirective(self, ctx: OpenSCENARIO2Parser.DoDirectiveContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#doDirective.
    def exitDoDirective(self, ctx: OpenSCENARIO2Parser.DoDirectiveContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#doMember.
    def enterDoMember(self, ctx: OpenSCENARIO2Parser.DoMemberContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#doMember.
    def exitDoMember(self, ctx: OpenSCENARIO2Parser.DoMemberContext):
        pass

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
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#behaviorInvocation.
    def exitBehaviorInvocation(
        self, ctx: OpenSCENARIO2Parser.BehaviorInvocationContext
    ):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#behaviorWithDeclaration.
    def enterBehaviorWithDeclaration(
        self, ctx: OpenSCENARIO2Parser.BehaviorWithDeclarationContext
    ):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#behaviorWithDeclaration.
    def exitBehaviorWithDeclaration(
        self, ctx: OpenSCENARIO2Parser.BehaviorWithDeclarationContext
    ):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#behaviorWithMember.
    def enterBehaviorWithMember(
        self, ctx: OpenSCENARIO2Parser.BehaviorWithMemberContext
    ):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#behaviorWithMember.
    def exitBehaviorWithMember(
        self, ctx: OpenSCENARIO2Parser.BehaviorWithMemberContext
    ):
        pass

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
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#waitDirective.
    def exitWaitDirective(self, ctx: OpenSCENARIO2Parser.WaitDirectiveContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#emitDirective.
    def enterEmitDirective(self, ctx: OpenSCENARIO2Parser.EmitDirectiveContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#emitDirective.
    def exitEmitDirective(self, ctx: OpenSCENARIO2Parser.EmitDirectiveContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#callDirective.
    def enterCallDirective(self, ctx: OpenSCENARIO2Parser.CallDirectiveContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#callDirective.
    def exitCallDirective(self, ctx: OpenSCENARIO2Parser.CallDirectiveContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#untilDirective.
    def enterUntilDirective(self, ctx: OpenSCENARIO2Parser.UntilDirectiveContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#untilDirective.
    def exitUntilDirective(self, ctx: OpenSCENARIO2Parser.UntilDirectiveContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#methodInvocation.
    def enterMethodInvocation(self, ctx: OpenSCENARIO2Parser.MethodInvocationContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#methodInvocation.
    def exitMethodInvocation(self, ctx: OpenSCENARIO2Parser.MethodInvocationContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#methodDeclaration.
    def enterMethodDeclaration(self, ctx: OpenSCENARIO2Parser.MethodDeclarationContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#methodDeclaration.
    def exitMethodDeclaration(self, ctx: OpenSCENARIO2Parser.MethodDeclarationContext):
        pass

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
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#methodImplementation.
    def exitMethodImplementation(
        self, ctx: OpenSCENARIO2Parser.MethodImplementationContext
    ):
        pass

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
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#coverDeclaration.
    def exitCoverDeclaration(self, ctx: OpenSCENARIO2Parser.CoverDeclarationContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#recordDeclaration.
    def enterRecordDeclaration(self, ctx: OpenSCENARIO2Parser.RecordDeclarationContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#recordDeclaration.
    def exitRecordDeclaration(self, ctx: OpenSCENARIO2Parser.RecordDeclarationContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#coverageExpression.
    def enterCoverageExpression(
        self, ctx: OpenSCENARIO2Parser.CoverageExpressionContext
    ):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#coverageExpression.
    def exitCoverageExpression(
        self, ctx: OpenSCENARIO2Parser.CoverageExpressionContext
    ):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#coverageUnit.
    def enterCoverageUnit(self, ctx: OpenSCENARIO2Parser.CoverageUnitContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#coverageUnit.
    def exitCoverageUnit(self, ctx: OpenSCENARIO2Parser.CoverageUnitContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#coverageRange.
    def enterCoverageRange(self, ctx: OpenSCENARIO2Parser.CoverageRangeContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#coverageRange.
    def exitCoverageRange(self, ctx: OpenSCENARIO2Parser.CoverageRangeContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#coverageEvery.
    def enterCoverageEvery(self, ctx: OpenSCENARIO2Parser.CoverageEveryContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#coverageEvery.
    def exitCoverageEvery(self, ctx: OpenSCENARIO2Parser.CoverageEveryContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#coverageEvent.
    def enterCoverageEvent(self, ctx: OpenSCENARIO2Parser.CoverageEventContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#coverageEvent.
    def exitCoverageEvent(self, ctx: OpenSCENARIO2Parser.CoverageEventContext):
        pass

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
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#implication.
    def exitImplication(self, ctx: OpenSCENARIO2Parser.ImplicationContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#disjunction.
    def enterDisjunction(self, ctx: OpenSCENARIO2Parser.DisjunctionContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#disjunction.
    def exitDisjunction(self, ctx: OpenSCENARIO2Parser.DisjunctionContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#conjunction.
    def enterConjunction(self, ctx: OpenSCENARIO2Parser.ConjunctionContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#conjunction.
    def exitConjunction(self, ctx: OpenSCENARIO2Parser.ConjunctionContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#inversion.
    def enterInversion(self, ctx: OpenSCENARIO2Parser.InversionContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#inversion.
    def exitInversion(self, ctx: OpenSCENARIO2Parser.InversionContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#relationExp.
    def enterRelationExp(self, ctx: OpenSCENARIO2Parser.RelationExpContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#relationExp.
    def exitRelationExp(self, ctx: OpenSCENARIO2Parser.RelationExpContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#sumExp.
    def enterSumExp(self, ctx: OpenSCENARIO2Parser.SumExpContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#sumExp.
    def exitSumExp(self, ctx: OpenSCENARIO2Parser.SumExpContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#relationalOp.
    def enterRelationalOp(self, ctx: OpenSCENARIO2Parser.RelationalOpContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#relationalOp.
    def exitRelationalOp(self, ctx: OpenSCENARIO2Parser.RelationalOpContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#termExp.
    def enterTermExp(self, ctx: OpenSCENARIO2Parser.TermExpContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#termExp.
    def exitTermExp(self, ctx: OpenSCENARIO2Parser.TermExpContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#additiveExp.
    def enterAdditiveExp(self, ctx: OpenSCENARIO2Parser.AdditiveExpContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#additiveExp.
    def exitAdditiveExp(self, ctx: OpenSCENARIO2Parser.AdditiveExpContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#additiveOp.
    def enterAdditiveOp(self, ctx: OpenSCENARIO2Parser.AdditiveOpContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#additiveOp.
    def exitAdditiveOp(self, ctx: OpenSCENARIO2Parser.AdditiveOpContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#multiplicativeExp.
    def enterMultiplicativeExp(self, ctx: OpenSCENARIO2Parser.MultiplicativeExpContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#multiplicativeExp.
    def exitMultiplicativeExp(self, ctx: OpenSCENARIO2Parser.MultiplicativeExpContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#factorExp.
    def enterFactorExp(self, ctx: OpenSCENARIO2Parser.FactorExpContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#factorExp.
    def exitFactorExp(self, ctx: OpenSCENARIO2Parser.FactorExpContext):
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
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#castExpression.
    def exitCastExpression(self, ctx: OpenSCENARIO2Parser.CastExpressionContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#functionApplicationExpression.
    def enterFunctionApplicationExpression(
        self, ctx: OpenSCENARIO2Parser.FunctionApplicationExpressionContext
    ):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#functionApplicationExpression.
    def exitFunctionApplicationExpression(
        self, ctx: OpenSCENARIO2Parser.FunctionApplicationExpressionContext
    ):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#fieldAccessExpression.
    def enterFieldAccessExpression(
        self, ctx: OpenSCENARIO2Parser.FieldAccessExpressionContext
    ):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#fieldAccessExpression.
    def exitFieldAccessExpression(
        self, ctx: OpenSCENARIO2Parser.FieldAccessExpressionContext
    ):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#elementAccessExpression.
    def enterElementAccessExpression(
        self, ctx: OpenSCENARIO2Parser.ElementAccessExpressionContext
    ):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#elementAccessExpression.
    def exitElementAccessExpression(
        self, ctx: OpenSCENARIO2Parser.ElementAccessExpressionContext
    ):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#typeTestExpression.
    def enterTypeTestExpression(
        self, ctx: OpenSCENARIO2Parser.TypeTestExpressionContext
    ):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#typeTestExpression.
    def exitTypeTestExpression(
        self, ctx: OpenSCENARIO2Parser.TypeTestExpressionContext
    ):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#fieldAccess.
    def enterFieldAccess(self, ctx: OpenSCENARIO2Parser.FieldAccessContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#fieldAccess.
    def exitFieldAccess(self, ctx: OpenSCENARIO2Parser.FieldAccessContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#primaryExp.
    def enterPrimaryExp(self, ctx: OpenSCENARIO2Parser.PrimaryExpContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#primaryExp.
    def exitPrimaryExp(self, ctx: OpenSCENARIO2Parser.PrimaryExpContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#valueExp.
    def enterValueExp(self, ctx: OpenSCENARIO2Parser.ValueExpContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#valueExp.
    def exitValueExp(self, ctx: OpenSCENARIO2Parser.ValueExpContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#listConstructor.
    def enterListConstructor(self, ctx: OpenSCENARIO2Parser.ListConstructorContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#listConstructor.
    def exitListConstructor(self, ctx: OpenSCENARIO2Parser.ListConstructorContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#rangeConstructor.
    def enterRangeConstructor(self, ctx: OpenSCENARIO2Parser.RangeConstructorContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#rangeConstructor.
    def exitRangeConstructor(self, ctx: OpenSCENARIO2Parser.RangeConstructorContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#identifierReference.
    def enterIdentifierReference(
        self, ctx: OpenSCENARIO2Parser.IdentifierReferenceContext
    ):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#identifierReference.
    def exitIdentifierReference(
        self, ctx: OpenSCENARIO2Parser.IdentifierReferenceContext
    ):
        pass

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
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#argumentSpecification.
    def exitArgumentSpecification(
        self, ctx: OpenSCENARIO2Parser.ArgumentSpecificationContext
    ):
        pass

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
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#positionalArgument.
    def exitPositionalArgument(
        self, ctx: OpenSCENARIO2Parser.PositionalArgumentContext
    ):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#namedArgument.
    def enterNamedArgument(self, ctx: OpenSCENARIO2Parser.NamedArgumentContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#namedArgument.
    def exitNamedArgument(self, ctx: OpenSCENARIO2Parser.NamedArgumentContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#physicalLiteral.
    def enterPhysicalLiteral(self, ctx: OpenSCENARIO2Parser.PhysicalLiteralContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#physicalLiteral.
    def exitPhysicalLiteral(self, ctx: OpenSCENARIO2Parser.PhysicalLiteralContext):
        pass

    # Enter a parse tree produced by OpenSCENARIO2Parser#integerLiteral.
    def enterIntegerLiteral(self, ctx: OpenSCENARIO2Parser.IntegerLiteralContext):
        pass

    # Exit a parse tree produced by OpenSCENARIO2Parser#integerLiteral.
    def exitIntegerLiteral(self, ctx: OpenSCENARIO2Parser.IntegerLiteralContext):
        pass


del OpenSCENARIO2Parser
