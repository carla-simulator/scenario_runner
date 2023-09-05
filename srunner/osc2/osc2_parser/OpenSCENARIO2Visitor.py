# Generated from .\OpenSCENARIO2.g4 by ANTLR 4.10.1
from antlr4 import *

if __name__ is not None and "." in __name__:
    from .OpenSCENARIO2Parser import OpenSCENARIO2Parser
else:
    from OpenSCENARIO2Parser import OpenSCENARIO2Parser


# This class defines a complete generic visitor for a parse tree produced by OpenSCENARIO2Parser.


class OpenSCENARIO2Visitor(ParseTreeVisitor):
    # Visit a parse tree produced by OpenSCENARIO2Parser#osc_file.
    def visitOsc_file(self, ctx: OpenSCENARIO2Parser.Osc_fileContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#preludeStatement.
    def visitPreludeStatement(self, ctx: OpenSCENARIO2Parser.PreludeStatementContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#importStatement.
    def visitImportStatement(self, ctx: OpenSCENARIO2Parser.ImportStatementContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#importReference.
    def visitImportReference(self, ctx: OpenSCENARIO2Parser.ImportReferenceContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#structuredIdentifier.
    def visitStructuredIdentifier(
        self, ctx: OpenSCENARIO2Parser.StructuredIdentifierContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#oscDeclaration.
    def visitOscDeclaration(self, ctx: OpenSCENARIO2Parser.OscDeclarationContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#physicalTypeDeclaration.
    def visitPhysicalTypeDeclaration(
        self, ctx: OpenSCENARIO2Parser.PhysicalTypeDeclarationContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#physicalTypeName.
    def visitPhysicalTypeName(self, ctx: OpenSCENARIO2Parser.PhysicalTypeNameContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#baseUnitSpecifier.
    def visitBaseUnitSpecifier(self, ctx: OpenSCENARIO2Parser.BaseUnitSpecifierContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#sIBaseUnitSpecifier.
    def visitSIBaseUnitSpecifier(
        self, ctx: OpenSCENARIO2Parser.SIBaseUnitSpecifierContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#unitDeclaration.
    def visitUnitDeclaration(self, ctx: OpenSCENARIO2Parser.UnitDeclarationContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#unitSpecifier.
    def visitUnitSpecifier(self, ctx: OpenSCENARIO2Parser.UnitSpecifierContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#sIUnitSpecifier.
    def visitSIUnitSpecifier(self, ctx: OpenSCENARIO2Parser.SIUnitSpecifierContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#sIBaseExponentList.
    def visitSIBaseExponentList(
        self, ctx: OpenSCENARIO2Parser.SIBaseExponentListContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#sIBaseExponent.
    def visitSIBaseExponent(self, ctx: OpenSCENARIO2Parser.SIBaseExponentContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#sIFactor.
    def visitSIFactor(self, ctx: OpenSCENARIO2Parser.SIFactorContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#sIOffset.
    def visitSIOffset(self, ctx: OpenSCENARIO2Parser.SIOffsetContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#enumDeclaration.
    def visitEnumDeclaration(self, ctx: OpenSCENARIO2Parser.EnumDeclarationContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#enumMemberDecl.
    def visitEnumMemberDecl(self, ctx: OpenSCENARIO2Parser.EnumMemberDeclContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#enumMemberValue.
    def visitEnumMemberValue(self, ctx: OpenSCENARIO2Parser.EnumMemberValueContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#enumName.
    def visitEnumName(self, ctx: OpenSCENARIO2Parser.EnumNameContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#enumMemberName.
    def visitEnumMemberName(self, ctx: OpenSCENARIO2Parser.EnumMemberNameContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#enumValueReference.
    def visitEnumValueReference(
        self, ctx: OpenSCENARIO2Parser.EnumValueReferenceContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#inheritsCondition.
    def visitInheritsCondition(self, ctx: OpenSCENARIO2Parser.InheritsConditionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#structDeclaration.
    def visitStructDeclaration(self, ctx: OpenSCENARIO2Parser.StructDeclarationContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#structInherts.
    def visitStructInherts(self, ctx: OpenSCENARIO2Parser.StructInhertsContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#structMemberDecl.
    def visitStructMemberDecl(self, ctx: OpenSCENARIO2Parser.StructMemberDeclContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#fieldName.
    def visitFieldName(self, ctx: OpenSCENARIO2Parser.FieldNameContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#structName.
    def visitStructName(self, ctx: OpenSCENARIO2Parser.StructNameContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#actorDeclaration.
    def visitActorDeclaration(self, ctx: OpenSCENARIO2Parser.ActorDeclarationContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#actorInherts.
    def visitActorInherts(self, ctx: OpenSCENARIO2Parser.ActorInhertsContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#actorMemberDecl.
    def visitActorMemberDecl(self, ctx: OpenSCENARIO2Parser.ActorMemberDeclContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#actorName.
    def visitActorName(self, ctx: OpenSCENARIO2Parser.ActorNameContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#scenarioDeclaration.
    def visitScenarioDeclaration(
        self, ctx: OpenSCENARIO2Parser.ScenarioDeclarationContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#scenarioInherts.
    def visitScenarioInherts(self, ctx: OpenSCENARIO2Parser.ScenarioInhertsContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#scenarioMemberDecl.
    def visitScenarioMemberDecl(
        self, ctx: OpenSCENARIO2Parser.ScenarioMemberDeclContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#qualifiedBehaviorName.
    def visitQualifiedBehaviorName(
        self, ctx: OpenSCENARIO2Parser.QualifiedBehaviorNameContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#behaviorName.
    def visitBehaviorName(self, ctx: OpenSCENARIO2Parser.BehaviorNameContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#actionDeclaration.
    def visitActionDeclaration(self, ctx: OpenSCENARIO2Parser.ActionDeclarationContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#actionInherts.
    def visitActionInherts(self, ctx: OpenSCENARIO2Parser.ActionInhertsContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#modifierDeclaration.
    def visitModifierDeclaration(
        self, ctx: OpenSCENARIO2Parser.ModifierDeclarationContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#modifierName.
    def visitModifierName(self, ctx: OpenSCENARIO2Parser.ModifierNameContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#typeExtension.
    def visitTypeExtension(self, ctx: OpenSCENARIO2Parser.TypeExtensionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#enumTypeExtension.
    def visitEnumTypeExtension(self, ctx: OpenSCENARIO2Parser.EnumTypeExtensionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#structuredTypeExtension.
    def visitStructuredTypeExtension(
        self, ctx: OpenSCENARIO2Parser.StructuredTypeExtensionContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#extendableTypeName.
    def visitExtendableTypeName(
        self, ctx: OpenSCENARIO2Parser.ExtendableTypeNameContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#extensionMemberDecl.
    def visitExtensionMemberDecl(
        self, ctx: OpenSCENARIO2Parser.ExtensionMemberDeclContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#globalParameterDeclaration.
    def visitGlobalParameterDeclaration(
        self, ctx: OpenSCENARIO2Parser.GlobalParameterDeclarationContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#typeDeclarator.
    def visitTypeDeclarator(self, ctx: OpenSCENARIO2Parser.TypeDeclaratorContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#nonAggregateTypeDeclarator.
    def visitNonAggregateTypeDeclarator(
        self, ctx: OpenSCENARIO2Parser.NonAggregateTypeDeclaratorContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#aggregateTypeDeclarator.
    def visitAggregateTypeDeclarator(
        self, ctx: OpenSCENARIO2Parser.AggregateTypeDeclaratorContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#listTypeDeclarator.
    def visitListTypeDeclarator(
        self, ctx: OpenSCENARIO2Parser.ListTypeDeclaratorContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#primitiveType.
    def visitPrimitiveType(self, ctx: OpenSCENARIO2Parser.PrimitiveTypeContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#typeName.
    def visitTypeName(self, ctx: OpenSCENARIO2Parser.TypeNameContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#eventDeclaration.
    def visitEventDeclaration(self, ctx: OpenSCENARIO2Parser.EventDeclarationContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#eventSpecification.
    def visitEventSpecification(
        self, ctx: OpenSCENARIO2Parser.EventSpecificationContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#eventReference.
    def visitEventReference(self, ctx: OpenSCENARIO2Parser.EventReferenceContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#eventFieldDecl.
    def visitEventFieldDecl(self, ctx: OpenSCENARIO2Parser.EventFieldDeclContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#eventFieldName.
    def visitEventFieldName(self, ctx: OpenSCENARIO2Parser.EventFieldNameContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#eventName.
    def visitEventName(self, ctx: OpenSCENARIO2Parser.EventNameContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#eventPath.
    def visitEventPath(self, ctx: OpenSCENARIO2Parser.EventPathContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#eventCondition.
    def visitEventCondition(self, ctx: OpenSCENARIO2Parser.EventConditionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#riseExpression.
    def visitRiseExpression(self, ctx: OpenSCENARIO2Parser.RiseExpressionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#fallExpression.
    def visitFallExpression(self, ctx: OpenSCENARIO2Parser.FallExpressionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#elapsedExpression.
    def visitElapsedExpression(self, ctx: OpenSCENARIO2Parser.ElapsedExpressionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#everyExpression.
    def visitEveryExpression(self, ctx: OpenSCENARIO2Parser.EveryExpressionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#boolExpression.
    def visitBoolExpression(self, ctx: OpenSCENARIO2Parser.BoolExpressionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#durationExpression.
    def visitDurationExpression(
        self, ctx: OpenSCENARIO2Parser.DurationExpressionContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#fieldDeclaration.
    def visitFieldDeclaration(self, ctx: OpenSCENARIO2Parser.FieldDeclarationContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#parameterDeclaration.
    def visitParameterDeclaration(
        self, ctx: OpenSCENARIO2Parser.ParameterDeclarationContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#variableDeclaration.
    def visitVariableDeclaration(
        self, ctx: OpenSCENARIO2Parser.VariableDeclarationContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#sampleExpression.
    def visitSampleExpression(self, ctx: OpenSCENARIO2Parser.SampleExpressionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#defaultValue.
    def visitDefaultValue(self, ctx: OpenSCENARIO2Parser.DefaultValueContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#parameterWithDeclaration.
    def visitParameterWithDeclaration(
        self, ctx: OpenSCENARIO2Parser.ParameterWithDeclarationContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#parameterWithMember.
    def visitParameterWithMember(
        self, ctx: OpenSCENARIO2Parser.ParameterWithMemberContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#constraintDeclaration.
    def visitConstraintDeclaration(
        self, ctx: OpenSCENARIO2Parser.ConstraintDeclarationContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#keepConstraintDeclaration.
    def visitKeepConstraintDeclaration(
        self, ctx: OpenSCENARIO2Parser.KeepConstraintDeclarationContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#constraintQualifier.
    def visitConstraintQualifier(
        self, ctx: OpenSCENARIO2Parser.ConstraintQualifierContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#constraintExpression.
    def visitConstraintExpression(
        self, ctx: OpenSCENARIO2Parser.ConstraintExpressionContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#removeDefaultDeclaration.
    def visitRemoveDefaultDeclaration(
        self, ctx: OpenSCENARIO2Parser.RemoveDefaultDeclarationContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#parameterReference.
    def visitParameterReference(
        self, ctx: OpenSCENARIO2Parser.ParameterReferenceContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#modifierInvocation.
    def visitModifierInvocation(
        self, ctx: OpenSCENARIO2Parser.ModifierInvocationContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#behaviorExpression.
    def visitBehaviorExpression(
        self, ctx: OpenSCENARIO2Parser.BehaviorExpressionContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#behaviorSpecification.
    def visitBehaviorSpecification(
        self, ctx: OpenSCENARIO2Parser.BehaviorSpecificationContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#onDirective.
    def visitOnDirective(self, ctx: OpenSCENARIO2Parser.OnDirectiveContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#onMember.
    def visitOnMember(self, ctx: OpenSCENARIO2Parser.OnMemberContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#doDirective.
    def visitDoDirective(self, ctx: OpenSCENARIO2Parser.DoDirectiveContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#doMember.
    def visitDoMember(self, ctx: OpenSCENARIO2Parser.DoMemberContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#composition.
    def visitComposition(self, ctx: OpenSCENARIO2Parser.CompositionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#compositionOperator.
    def visitCompositionOperator(
        self, ctx: OpenSCENARIO2Parser.CompositionOperatorContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#behaviorInvocation.
    def visitBehaviorInvocation(
        self, ctx: OpenSCENARIO2Parser.BehaviorInvocationContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#behaviorWithDeclaration.
    def visitBehaviorWithDeclaration(
        self, ctx: OpenSCENARIO2Parser.BehaviorWithDeclarationContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#behaviorWithMember.
    def visitBehaviorWithMember(
        self, ctx: OpenSCENARIO2Parser.BehaviorWithMemberContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#labelName.
    def visitLabelName(self, ctx: OpenSCENARIO2Parser.LabelNameContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#actorExpression.
    def visitActorExpression(self, ctx: OpenSCENARIO2Parser.ActorExpressionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#waitDirective.
    def visitWaitDirective(self, ctx: OpenSCENARIO2Parser.WaitDirectiveContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#emitDirective.
    def visitEmitDirective(self, ctx: OpenSCENARIO2Parser.EmitDirectiveContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#callDirective.
    def visitCallDirective(self, ctx: OpenSCENARIO2Parser.CallDirectiveContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#untilDirective.
    def visitUntilDirective(self, ctx: OpenSCENARIO2Parser.UntilDirectiveContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#methodInvocation.
    def visitMethodInvocation(self, ctx: OpenSCENARIO2Parser.MethodInvocationContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#methodDeclaration.
    def visitMethodDeclaration(self, ctx: OpenSCENARIO2Parser.MethodDeclarationContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#returnType.
    def visitReturnType(self, ctx: OpenSCENARIO2Parser.ReturnTypeContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#methodImplementation.
    def visitMethodImplementation(
        self, ctx: OpenSCENARIO2Parser.MethodImplementationContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#methodQualifier.
    def visitMethodQualifier(self, ctx: OpenSCENARIO2Parser.MethodQualifierContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#methodName.
    def visitMethodName(self, ctx: OpenSCENARIO2Parser.MethodNameContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#coverageDeclaration.
    def visitCoverageDeclaration(
        self, ctx: OpenSCENARIO2Parser.CoverageDeclarationContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#coverDeclaration.
    def visitCoverDeclaration(self, ctx: OpenSCENARIO2Parser.CoverDeclarationContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#recordDeclaration.
    def visitRecordDeclaration(self, ctx: OpenSCENARIO2Parser.RecordDeclarationContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#coverageExpression.
    def visitCoverageExpression(
        self, ctx: OpenSCENARIO2Parser.CoverageExpressionContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#coverageUnit.
    def visitCoverageUnit(self, ctx: OpenSCENARIO2Parser.CoverageUnitContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#coverageRange.
    def visitCoverageRange(self, ctx: OpenSCENARIO2Parser.CoverageRangeContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#coverageEvery.
    def visitCoverageEvery(self, ctx: OpenSCENARIO2Parser.CoverageEveryContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#coverageEvent.
    def visitCoverageEvent(self, ctx: OpenSCENARIO2Parser.CoverageEventContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#coverageNameArgument.
    def visitCoverageNameArgument(
        self, ctx: OpenSCENARIO2Parser.CoverageNameArgumentContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#targetName.
    def visitTargetName(self, ctx: OpenSCENARIO2Parser.TargetNameContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#expression.
    def visitExpression(self, ctx: OpenSCENARIO2Parser.ExpressionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#ternaryOpExp.
    def visitTernaryOpExp(self, ctx: OpenSCENARIO2Parser.TernaryOpExpContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#implication.
    def visitImplication(self, ctx: OpenSCENARIO2Parser.ImplicationContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#disjunction.
    def visitDisjunction(self, ctx: OpenSCENARIO2Parser.DisjunctionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#conjunction.
    def visitConjunction(self, ctx: OpenSCENARIO2Parser.ConjunctionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#inversion.
    def visitInversion(self, ctx: OpenSCENARIO2Parser.InversionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#relationExp.
    def visitRelationExp(self, ctx: OpenSCENARIO2Parser.RelationExpContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#sumExp.
    def visitSumExp(self, ctx: OpenSCENARIO2Parser.SumExpContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#relationalOp.
    def visitRelationalOp(self, ctx: OpenSCENARIO2Parser.RelationalOpContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#termExp.
    def visitTermExp(self, ctx: OpenSCENARIO2Parser.TermExpContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#additiveExp.
    def visitAdditiveExp(self, ctx: OpenSCENARIO2Parser.AdditiveExpContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#additiveOp.
    def visitAdditiveOp(self, ctx: OpenSCENARIO2Parser.AdditiveOpContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#multiplicativeExp.
    def visitMultiplicativeExp(self, ctx: OpenSCENARIO2Parser.MultiplicativeExpContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#factorExp.
    def visitFactorExp(self, ctx: OpenSCENARIO2Parser.FactorExpContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#multiplicativeOp.
    def visitMultiplicativeOp(self, ctx: OpenSCENARIO2Parser.MultiplicativeOpContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#factor.
    def visitFactor(self, ctx: OpenSCENARIO2Parser.FactorContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#primaryExpression.
    def visitPrimaryExpression(self, ctx: OpenSCENARIO2Parser.PrimaryExpressionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#castExpression.
    def visitCastExpression(self, ctx: OpenSCENARIO2Parser.CastExpressionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#functionApplicationExpression.
    def visitFunctionApplicationExpression(
        self, ctx: OpenSCENARIO2Parser.FunctionApplicationExpressionContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#fieldAccessExpression.
    def visitFieldAccessExpression(
        self, ctx: OpenSCENARIO2Parser.FieldAccessExpressionContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#elementAccessExpression.
    def visitElementAccessExpression(
        self, ctx: OpenSCENARIO2Parser.ElementAccessExpressionContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#typeTestExpression.
    def visitTypeTestExpression(
        self, ctx: OpenSCENARIO2Parser.TypeTestExpressionContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#fieldAccess.
    def visitFieldAccess(self, ctx: OpenSCENARIO2Parser.FieldAccessContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#primaryExp.
    def visitPrimaryExp(self, ctx: OpenSCENARIO2Parser.PrimaryExpContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#valueExp.
    def visitValueExp(self, ctx: OpenSCENARIO2Parser.ValueExpContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#listConstructor.
    def visitListConstructor(self, ctx: OpenSCENARIO2Parser.ListConstructorContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#rangeConstructor.
    def visitRangeConstructor(self, ctx: OpenSCENARIO2Parser.RangeConstructorContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#identifierReference.
    def visitIdentifierReference(
        self, ctx: OpenSCENARIO2Parser.IdentifierReferenceContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#argumentListSpecification.
    def visitArgumentListSpecification(
        self, ctx: OpenSCENARIO2Parser.ArgumentListSpecificationContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#argumentSpecification.
    def visitArgumentSpecification(
        self, ctx: OpenSCENARIO2Parser.ArgumentSpecificationContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#argumentName.
    def visitArgumentName(self, ctx: OpenSCENARIO2Parser.ArgumentNameContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#argumentList.
    def visitArgumentList(self, ctx: OpenSCENARIO2Parser.ArgumentListContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#positionalArgument.
    def visitPositionalArgument(
        self, ctx: OpenSCENARIO2Parser.PositionalArgumentContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#namedArgument.
    def visitNamedArgument(self, ctx: OpenSCENARIO2Parser.NamedArgumentContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#physicalLiteral.
    def visitPhysicalLiteral(self, ctx: OpenSCENARIO2Parser.PhysicalLiteralContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by OpenSCENARIO2Parser#integerLiteral.
    def visitIntegerLiteral(self, ctx: OpenSCENARIO2Parser.IntegerLiteralContext):
        return self.visitChildren(ctx)


del OpenSCENARIO2Parser
