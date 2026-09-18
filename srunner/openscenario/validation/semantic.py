"""Small, extensible semantic validation boundary."""

from __future__ import absolute_import

from srunner.openscenario.validation.diagnostics import Diagnostic, ValidationError
from srunner.openscenario.model.references import Position


class SemanticValidator(object):
    """Validate invariants that are not expressed by the XML schema."""

    def validate(self, document, raise_on_error=True):
        diagnostics = []
        names = set()
        for declaration in document.parameter_declarations:
            if not declaration.name:
                diagnostics.append(Diagnostic(
                    "semantic_error", "ParameterDeclaration requires a name", source=declaration.source))
            elif declaration.name in names:
                diagnostics.append(Diagnostic(
                    "semantic_error", "Duplicate parameter declaration {!r}".format(declaration.name),
                    source=declaration.source))
            names.add(declaration.name)

        variable_names = set()
        for declaration in getattr(document, "variable_declarations", []):
            if not declaration.name:
                diagnostics.append(Diagnostic("semantic_error", "VariableDeclaration requires a name",
                                               source=declaration.source))
            elif declaration.name in variable_names:
                diagnostics.append(Diagnostic("semantic_error", "Duplicate variable declaration {!r}".format(
                    declaration.name), source=declaration.source))
            variable_names.add(declaration.name)

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

        parameter_names = set(item.name for item in document.parameter_declarations)
        for action in document.parameter_actions():
            if action.parameter_ref not in parameter_names:
                diagnostics.append(Diagnostic("semantic_error", "Unknown parameter reference {!r}".format(
                    action.parameter_ref), source=action.source))

        for action in document.variable_actions():
            variable_name = (action.variable_ref or "").lstrip("$")
            if variable_name not in variable_names:
                diagnostics.append(Diagnostic("semantic_error", "Unknown variable reference {!r}".format(
                    action.variable_ref), source=action.source))
            if hasattr(action, "rule") and action.rule not in ("add", "subtract", "multiply", "divide", "+", "-", "*", "/"):
                diagnostics.append(Diagnostic("semantic_error", "Unsupported variable modification {!r}".format(
                    action.rule), source=action.source))

        if diagnostics and raise_on_error:
            raise ValidationError(diagnostics)
        return diagnostics
