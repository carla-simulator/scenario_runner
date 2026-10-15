"""Small, extensible semantic validation boundary."""

from __future__ import absolute_import

from srunner.openscenario.validation.diagnostics import Diagnostic, ValidationError
# Module import (not names): the expression module imports this package's diagnostics.
from srunner.openscenario.model import expressions
from srunner.openscenario.model.references import Position


class SemanticValidator(object):
    """Validate invariants that are not expressed by the XML schema."""

    def validate(self, document, raise_on_error=True):
        diagnostics = []
        diagnostics.extend(self._declarations(document, "parameter_declarations", "parameter_store",
                                              "ParameterDeclaration", "parameter"))
        diagnostics.extend(self._declarations(document, "variable_declarations", "variable_store",
                                              "VariableDeclaration", "variable"))

        for reference in getattr(document, "catalog_references", []):
            if not reference.catalog_name or not reference.entry_name:
                diagnostics.append(Diagnostic("semantic_error", "CatalogReference requires catalogName and entryName",
                                               source=reference.source))

        if hasattr(document, "positions"):
            for element in document.root.iter("Position"):
                try:
                    position = Position.from_xml(element)
                    if position is not None and position.kind == "RoutePosition" and not position.attributes.get("route_ref"):
                        diagnostics.append(Diagnostic("semantic_error", "RoutePosition requires a route reference",
                                                       source=position.source))
                except ValueError as error:
                    diagnostics.append(Diagnostic("semantic_error", str(error), source=element))
                    break

        if hasattr(document, "variable_actions"):
            diagnostics.extend(self._references(document))

        if diagnostics and raise_on_error:
            raise ValidationError(diagnostics)
        return diagnostics

    @staticmethod
    def _declarations(document, attribute, store_attribute, element_name, kind):
        diagnostics = []
        declarations = getattr(document, attribute, [])
        for declaration in declarations:
            if not declaration.name:
                diagnostics.append(Diagnostic("semantic_error", "{} requires a name".format(element_name),
                                              source=declaration.source))
        store = getattr(document, store_attribute, None)
        if store is not None:
            return diagnostics + list(store.diagnostics)
        names = set()
        for declaration in declarations:
            if declaration.name and declaration.name in names:
                diagnostics.append(Diagnostic(
                    "semantic_error", "Duplicate {} declaration {!r}".format(kind, declaration.name),
                    source=declaration.source, code=Diagnostic.DUPLICATE_DECLARATION))
            names.add(declaration.name)
        return diagnostics

    # Attributes that name a parameter/variable instead of using its value.
    _NAME_ATTRIBUTES = {
        "ParameterDeclaration": ("name",),
        "VariableDeclaration": ("name",),
        "ParameterAssignment": ("parameterRef",),
        "ParameterAction": ("parameterRef",),
        "ParameterCondition": ("parameterRef",),
        "VariableAction": ("variableRef",),
        "VariableCondition": ("variableRef",),
    }

    def _references(self, document):
        """Check parameter/variable references.

        Every attribute value in the document is scanned for ``$name`` and
        ``${...}`` references.  Scoping is not modelled: a parameter declared in
        any ParameterDeclarations of the document (including nested catalog
        entries or stories) is an accepted reference target everywhere.
        """
        diagnostics = []
        parameters = set(item.name for item in document.parameter_declarations)
        parameters.update(element.attrib.get("name") for element in document.root.iter("ParameterDeclaration"))
        variables = dict((item.name, item.parameter_type) for item in document.variable_declarations)

        for action in document.parameter_actions(include_story=True):
            if (action.parameter_ref or "").lstrip("$") not in parameters:
                diagnostics.append(_unknown("parameter", action.parameter_ref, action.source))

        for action in document.variable_actions(include_story=True):
            name = (action.variable_ref or "").lstrip("$")
            if name not in variables:
                diagnostics.append(_unknown("variable", action.variable_ref, action.source))
            elif hasattr(action, "rule") and variables[name] not in expressions.NUMERIC_TYPES:
                diagnostics.append(Diagnostic(
                    "semantic_error", "VariableModifyAction cannot modify {} variable {!r}".format(
                        variables[name], name), source=action.source, code=Diagnostic.TYPE_MISMATCH))

        for element in document.value_conditions("VariableCondition"):
            name = (element.attrib.get("variableRef") or "").lstrip("$")
            rule = element.attrib.get("rule")
            if name not in variables:
                diagnostics.append(_unknown("variable", element.attrib.get("variableRef"), element))
            elif variables[name] in ("string", "dateTime") and rule not in ("equalTo", "notEqualTo"):
                diagnostics.append(Diagnostic(
                    "semantic_error", "Rule {!r} is not defined for {} variable {!r}".format(
                        rule, variables[name], name), source=element, code=Diagnostic.TYPE_MISMATCH))

        for condition in document.root.iter("Condition"):
            # ScenarioRunner criteria_* conditions use parameterRef to name a criterion argument.
            if (condition.attrib.get("name") or "").startswith("criteria_"):
                continue
            for element in condition.iter("ParameterCondition"):
                name = (element.attrib.get("parameterRef") or "").lstrip("$")
                if name not in parameters:
                    diagnostics.append(_unknown("parameter", element.attrib.get("parameterRef"), element))

        for element in document.root.iter():
            skipped = self._NAME_ATTRIBUTES.get(element.tag, ())
            for key, value in element.attrib.items():
                if key not in skipped and "$" in value:
                    diagnostics.extend(_check_value(value, parameters, variables, element))
        return diagnostics


def _unknown(kind, name, source):
    return Diagnostic("semantic_error", "Unknown {} reference {!r}".format(kind, name), source=source,
                      code=Diagnostic.UNKNOWN_REFERENCE)


def _check_value(value, parameters, variables, source):
    """Check references and syntax of an attribute value without evaluating it."""
    try:
        if expressions.is_expression(value):
            names = expressions.check_expression(value)
        else:
            name = expressions.reference_name(value)
            if name is None and isinstance(value, str) and value.strip().startswith("$"):
                raise expressions.EvaluationError("Invalid parameter reference", Diagnostic.INVALID_EXPRESSION,
                                                  value)
            names = [name] if name is not None else []
    except expressions.EvaluationError as error:
        error.diagnostic.source = source
        return [error.diagnostic]
    diagnostics = []
    for name in names:
        if name in parameters:
            continue
        message = "Unknown parameter reference {!r} in {!r}".format(name, value)
        if name in variables:
            message += "; variables can only be read by a VariableCondition"
        diagnostics.append(Diagnostic("semantic_error", message, source=source, code=Diagnostic.UNKNOWN_REFERENCE))
    return diagnostics
