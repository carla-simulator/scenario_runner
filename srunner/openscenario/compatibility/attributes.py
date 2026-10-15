"""Resolve XML attribute values (literals, ``$name`` references, ``${...}`` expressions)."""

from __future__ import absolute_import

from srunner.openscenario.compatibility.global_parameters import GlobalParameterMapping
from srunner.openscenario.model.expressions import EvaluationError, is_expression
from srunner.openscenario.model.parameters import EvaluationContext, ParameterStore
from srunner.openscenario.validation.diagnostics import Diagnostic
from srunner.tools.util import strtobool


def _to_bool(value):
    if isinstance(value, bool):
        return value
    return bool(strtobool(str(value)))


def _to_string(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


# Literal conversions keep the lenient legacy parsing (e.g. int("1.0"), "yes" as boolean).
_CONVERTERS = {
    "double": float,
    "int": lambda value: int(float(value)),
    "boolean": _to_bool,
    "string": _to_string,
}


def global_evaluation():
    """Evaluation context reading the runtime's global parameter dictionary (untyped)."""
    return EvaluationContext(parameters=ParameterStore(binding=GlobalParameterMapping()))


def resolve_attribute(value, expected_type=None, evaluation=None, default=None):
    """Return ``value`` resolved and converted to ``expected_type`` (double, int, boolean, string).

    ``None`` yields ``default`` (converted as well).  Failures raise EvaluationError
    carrying a Diagnostic.
    """
    if value is None:
        if default is None:
            return None
        value = default
    text = value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("$"):
            evaluation = evaluation or global_evaluation()
            if is_expression(stripped):
                value = evaluation.resolve(stripped, expected_type)
            else:
                value = evaluation.resolve(stripped)
    if expected_type is None:
        return value
    try:
        return _CONVERTERS[expected_type](value)
    except (TypeError, ValueError) as error:
        raise EvaluationError("Cannot convert {!r} to {}".format(value, expected_type),
                              Diagnostic.TYPE_MISMATCH, text) from error
