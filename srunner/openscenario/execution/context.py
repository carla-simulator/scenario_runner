"""Execution context passed to semantic action executors."""

from __future__ import absolute_import

from srunner.openscenario.execution.capabilities import BackendCapabilities
from srunner.openscenario.model.expressions import EvaluationError, is_expression, reference_name
from srunner.openscenario.model.parameters import EvaluationContext, convert_value
from srunner.openscenario.validation.diagnostics import (
    Diagnostic, ValidationError, diagnostics_as_validation_error)
from srunner.openscenario.validation.semantic import SemanticValidator


class ExecutionContext(object):
    def __init__(self, backend=None, capabilities=None, state=None, document=None, diagnostics=None):
        self.backend = backend
        self.capabilities = capabilities or BackendCapabilities()
        self.state = state if state is not None else {}
        self.evaluation = self.state.get("evaluation") or EvaluationContext()
        self.state["evaluation"] = self.evaluation
        self.document = document
        self.diagnostics = list(diagnostics or [])

    @classmethod
    def for_document(cls, document, backend=None, capabilities=None, parameter_overrides=None, state=None,
                     parameter_binding=None, validator=None):
        """Create a context whose evaluation state is the document's parameter and variable stores.

        ``parameter_overrides`` are the user-supplied parameter values (for
        example ``--openscenarioparams``).  ``parameter_binding`` is an optional
        mutable mapping the parameter store mirrors (the runtime's global
        parameter dictionary).  Construction does not raise on document errors;
        they are collected in ``diagnostics`` (see :meth:`raise_for_errors`).
        An unexpected evaluation failure is raised as ValidationError.
        """
        with diagnostics_as_validation_error():
            return cls._for_document(document, backend, capabilities, parameter_overrides, state,
                                     parameter_binding, validator)

    @classmethod
    def _for_document(cls, document, backend, capabilities, parameter_overrides, state, parameter_binding,
                      validator):
        diagnostics = []
        parameters = document.parameter_store
        variables = document.variable_store
        evaluation = EvaluationContext(parameters=parameters, variables=variables)

        for name, value in (parameter_overrides or {}).items():
            if not parameters.is_declared(name):
                diagnostics.append(Diagnostic(
                    "semantic_error", "Override for undeclared parameter {!r} is ignored".format(name),
                    severity="warning", code=Diagnostic.UNDECLARED_OVERRIDE))
                continue
            try:
                parameters.set(name, value)
            except EvaluationError as error:
                diagnostics.append(error.diagnostic)

        overridden = set(parameter_overrides or {})
        for declaration in document.parameter_declarations:
            if declaration.name in overridden or not declaration.name:
                continue
            diagnostics.extend(_initialise(declaration, evaluation, parameters))
        for declaration in document.variable_declarations:
            if declaration.name:
                diagnostics.extend(_initialise(declaration, evaluation, variables))

        if parameter_binding is not None:
            parameters.bind(parameter_binding)

        validation = (validator or SemanticValidator()).validate(document, raise_on_error=False)
        # The validator also reports unknown references in declaration values; keep one diagnostic each.
        reported = set(id(item.source) for item in validation if item.code == Diagnostic.UNKNOWN_REFERENCE)
        diagnostics = [item for item in diagnostics
                       if item.code != Diagnostic.UNKNOWN_REFERENCE or id(item.source) not in reported]
        diagnostics.extend(validation)
        state = state if state is not None else {}
        state["evaluation"] = evaluation
        return cls(backend=backend, capabilities=capabilities, state=state, document=document,
                   diagnostics=diagnostics)

    @property
    def errors(self):
        return [diagnostic for diagnostic in self.diagnostics if diagnostic.is_error]

    def raise_for_errors(self):
        if self.errors:
            raise ValidationError(self.errors)


def _initialise(declaration, evaluation, store):
    """Evaluate a declaration's initial value (literal, reference or expression) into ``store``."""
    value = declaration.value
    try:
        if is_expression(value) or reference_name(value) is not None:
            store.set(declaration.name, evaluation.resolve(value, declaration.parameter_type))
        elif value is not None:
            # Literals keep their lexical form (as the legacy runtime stores them); only check the type.
            convert_value(value, declaration.parameter_type, "$" + declaration.name)
    except EvaluationError as error:
        diagnostic = error.diagnostic
        diagnostic.source = declaration.source
        return [diagnostic]
    return []
