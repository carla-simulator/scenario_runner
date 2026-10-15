"""Small, extensible semantic validation boundary."""

from __future__ import absolute_import

from srunner.openscenario.validation.diagnostics import Diagnostic, ValidationError


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

        if diagnostics and raise_on_error:
            raise ValidationError(diagnostics)
        return diagnostics
