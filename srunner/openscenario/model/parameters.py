"""Typed OpenSCENARIO values, parameters, variables, and evaluation state."""

from __future__ import absolute_import

import operator

from srunner.openscenario.model.expressions import (
    NUMERIC_TYPES, EvaluationError, UnknownReferenceError, check_integer_range, coerce, evaluate,
    is_expression, reference_name)
from srunner.openscenario.validation.diagnostics import Diagnostic


class ParameterScope(object):
    """Names the semantic scope in which a declaration is visible."""

    GLOBAL = "global"
    CATALOG_ENTRY = "catalog_entry"
    LOCAL = "local"


class ParameterDeclaration(object):
    """A parameter declaration without simulator-specific interpretation."""

    def __init__(self, name, parameter_type, value, scope=ParameterScope.GLOBAL, source=None):
        self.name = name
        self.parameter_type = parameter_type
        self.value = value
        self.scope = scope
        self.source = source

    def __repr__(self):
        return "ParameterDeclaration(name={!r}, type={!r}, value={!r})".format(
            self.name, self.parameter_type, self.value)


def _to_int(value):
    if isinstance(value, bool):
        raise ValueError("Boolean value {!r} is not an integer".format(value))
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError("Non-integral value {!r} for an integer type".format(value))
        return int(value)
    return int(value)


def _to_bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("true", "1"):
        return True
    if text in ("false", "0"):
        return False
    raise ValueError("Invalid boolean value {!r}".format(value))


def _unsigned_int(value):
    result = _to_int(value)
    if result < 0:
        raise ValueError("Unsigned value cannot be negative")
    return result


class ParameterValue(object):
    """A declared OSC value with its lexical representation retained."""

    _CONVERTERS = {
        "string": lambda value: value,
        "boolean": _to_bool,
        "int": _to_int,
        "integer": _to_int,
        "unsignedInt": _unsigned_int,
        "double": float,
        "float": float,
        "unsignedShort": _unsigned_int,
        "unsignedByte": _unsigned_int,
    }

    def __init__(self, value, value_type=None):
        self.raw = value
        self.value_type = value_type

    def convert(self, value_type=None):
        value_type = value_type or self.value_type
        converter = self._CONVERTERS.get(value_type, lambda value: value)
        return converter(self.raw)

    def __str__(self):
        return str(self.raw)


def convert_value(value, value_type, expression=None):
    """Convert a literal to ``value_type``; failures raise EvaluationError."""
    try:
        result = ParameterValue(value, value_type).convert()
    except (TypeError, ValueError) as error:
        if isinstance(error, EvaluationError):
            raise
        raise EvaluationError("Cannot convert {!r} to {}: {}".format(value, value_type, error),
                              Diagnostic.TYPE_MISMATCH, expression) from error
    if isinstance(result, int) and not isinstance(result, bool) and value_type:
        check_integer_range(result, value_type, expression)
    return result


def resolve_value(expression, lookup, expected_type=None):
    """Resolve a literal, ``$name`` reference or ``${...}`` expression.

    ``lookup(name)`` returns the typed value of a referenced parameter or raises
    UnknownReferenceError.  The result is converted to ``expected_type`` if given.
    """
    if isinstance(expression, ParameterValue):
        return convert_value(expression.raw, expected_type or expression.value_type)
    if not isinstance(expression, str):
        return convert_value(expression, expected_type) if expected_type else expression
    if is_expression(expression):
        return coerce(evaluate(expression, lookup), expected_type, expression)
    name = reference_name(expression)
    if name is not None:
        try:
            value = lookup(name)
        except KeyError as error:
            raise UnknownReferenceError(name, expression) from error
        return convert_value(value, expected_type, expression) if expected_type else value
    if expression.startswith("$"):
        raise EvaluationError("Invalid parameter reference", Diagnostic.INVALID_EXPRESSION, expression)
    return convert_value(expression, expected_type, expression) if expected_type else expression


def _plain(value):
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    return str(value)


def _reference_name(value):
    text = str(value)
    return text[1:] if text.startswith("$") else text


_COMPARISONS = {
    "equalTo": operator.eq,
    "notEqualTo": operator.ne,
    "greaterThan": operator.gt,
    "lessThan": operator.lt,
    "greaterOrEqual": operator.ge,
    "lessOrEqual": operator.le,
}


def compare(current, rule, reference, value_type=None, name=None):
    """Compare ``current`` with ``reference`` following the OSC ``Rule`` semantics."""
    if rule not in _COMPARISONS:
        raise EvaluationError("Unsupported comparison rule {!r}".format(rule), Diagnostic.INVALID_EXPRESSION)
    if value_type is None:
        current, reference = _comparable(current), _comparable(reference)
    is_text = isinstance(current, str) or isinstance(reference, str) or value_type in ("string", "dateTime")
    if is_text and rule not in ("equalTo", "notEqualTo"):
        raise EvaluationError("Rule {!r} is not defined for non-numeric value of '{}'".format(rule, name),
                              Diagnostic.TYPE_MISMATCH)
    if is_text:
        current, reference = str(current), str(reference)
    return _COMPARISONS[rule](current, reference)


def _comparable(value):
    value = _plain(value)
    if not isinstance(value, str):
        return value
    try:
        return float(value)
    except ValueError:
        return value


class ParameterStore(object):
    """Scoped, typed declaration store used by parameters and variables.

    A store may be bound to an external mutable mapping (for example the
    runtime's global parameter dictionary); writes go through to the mapping and
    reads prefer it, so both views never diverge.
    """

    kind = "parameter"
    _MODIFY_RULES = {"+": operator.add, "add": operator.add, "*": operator.mul, "multiply": operator.mul,
                     "-": operator.sub, "subtract": operator.sub}

    def __init__(self, declarations=None, values=None, binding=None):
        self._declarations = {}
        self._values = {}
        self._binding = None
        self.diagnostics = []
        for declaration in declarations or []:
            self.declare(declaration)
        for name, value in (values or {}).items():
            self.set(name, value)
        if binding is not None:
            self.bind(binding)

    def declare(self, declaration):
        # Construction never throws: duplicates are recorded so callers can
        # collect every document error before execution.  Last declaration wins.
        if declaration.name in self._declarations:
            self.diagnostics.append(Diagnostic(
                "semantic_error", "Duplicate {} declaration {!r}".format(self.kind, declaration.name),
                source=declaration.source, code=Diagnostic.DUPLICATE_DECLARATION))
        self._declarations[declaration.name] = declaration
        self._values[declaration.name] = ParameterValue(declaration.value, declaration.parameter_type)
        if self._binding is not None:
            self._binding[declaration.name] = declaration.value

    def bind(self, mapping):
        """Mirror this store into ``mapping``; current declared values are pushed into it."""
        self._binding = mapping
        for name, value in self._values.items():
            mapping[name] = value.raw
        return self

    @property
    def binding(self):
        return self._binding

    def names(self):
        return tuple(self._declarations)

    def is_declared(self, name):
        return _reference_name(name) in self._declarations

    def declaration(self, name):
        name = _reference_name(name)
        if name not in self._declarations:
            raise UnknownReferenceError(name)
        return self._declarations[name]

    def value_type(self, name):
        declaration = self._declarations.get(_reference_name(name))
        return declaration.parameter_type if declaration is not None else None

    def raw(self, name):
        name = _reference_name(name)
        if self._binding is not None and name in self._binding:
            return _plain(self._binding[name])
        if name not in self._values:
            raise UnknownReferenceError(name)
        return self._values[name].raw

    def get(self, name, expected_type=None):
        name = _reference_name(name)
        raw = self.raw(name)
        value_type = expected_type or self.value_type(name)
        return convert_value(raw, value_type, "$" + name) if value_type else raw

    def set(self, name, value):
        name = _reference_name(name)
        declaration = self.declaration(name)
        converted = convert_value(value, declaration.parameter_type, "$" + name)
        self._values[name] = ParameterValue(converted, declaration.parameter_type)
        if self._binding is not None:
            self._binding[name] = converted
        return converted

    def apply_modification(self, name, rule, operand):
        if rule not in self._MODIFY_RULES:
            raise EvaluationError("Unsupported {} modification rule {!r}".format(self.kind, rule),
                                  Diagnostic.INVALID_EXPRESSION)
        value_type = self.declaration(name).parameter_type
        if value_type not in NUMERIC_TYPES:
            raise EvaluationError("Cannot modify {} '{}' of type {}".format(self.kind, name, value_type),
                                  Diagnostic.TYPE_MISMATCH)
        try:
            result = self._MODIFY_RULES[rule](self.get(name), operand)
        except ZeroDivisionError as error:
            raise EvaluationError("Division by zero modifying '{}'".format(name),
                                  Diagnostic.ARITHMETIC_ERROR) from error
        return self.set(name, coerce(result, value_type, "$" + name))

    def modify(self, name, rule, value):
        """Apply an add/multiply rule; ``value`` may reference other parameters of this store."""
        name = _reference_name(name)
        return self.apply_modification(name, rule, self.resolve(value, self.declaration(name).parameter_type))

    def resolve(self, expression, expected_type=None):
        return resolve_value(expression, self.get, expected_type)


class VariableStore(ParameterStore):
    """Runtime variable namespace with explicit typed assignment semantics.

    OSC variables cannot be referenced from expressions, so modification
    operands are literals here; resolve parameter references through
    :class:`EvaluationContext` first.
    """

    kind = "variable"
    _MODIFY_RULES = dict(ParameterStore._MODIFY_RULES, **{"/": operator.truediv, "divide": operator.truediv})

    def modify(self, name, rule, value):
        name = _reference_name(name)
        operand = convert_value(value, self.declaration(name).parameter_type, "$" + name)
        return self.apply_modification(name, rule, operand)


class EvaluationContext(object):
    """Shared parameter/variable evaluation context for semantic execution.

    Expressions and ``$name`` references resolve against parameters only: per
    the standard, variables are read exclusively through VariableCondition.
    """

    def __init__(self, parameters=None, variables=None):
        self.parameters = parameters if parameters is not None else ParameterStore()
        self.variables = variables if variables is not None else VariableStore()

    def lookup(self, name):
        return self.parameters.get(name)

    def resolve(self, expression, expected_type=None):
        return resolve_value(expression, self.lookup, expected_type)

    def set_parameter(self, name, expression):
        return self.parameters.set(name, self.resolve(expression, self.parameters.declaration(name).parameter_type))

    def modify_parameter(self, name, rule, expression):
        operand = self.resolve(expression, self.parameters.declaration(name).parameter_type)
        return self.parameters.apply_modification(_reference_name(name), rule, operand)

    def set_variable(self, name, expression):
        return self.variables.set(name, self.resolve(expression, self.variables.declaration(name).parameter_type))

    def modify_variable(self, name, rule, expression):
        operand = self.resolve(expression, self.variables.declaration(name).parameter_type)
        return self.variables.modify(name, rule, operand)

    def compare_variable(self, name, rule, expression):
        value_type = self.variables.declaration(name).parameter_type
        return compare(self.variables.get(name), rule, self.resolve(expression, value_type), value_type, name)

    def compare_parameter(self, name, rule, expression):
        value_type = self.parameters.value_type(name)
        return compare(self.parameters.get(name), rule, self.resolve(expression, value_type), value_type, name)


class VariableDeclaration(ParameterDeclaration):
    """Placeholder semantic type for the distinct OSC variable namespace."""

    def __init__(self, name, variable_type, value, scope=ParameterScope.LOCAL, source=None):
        super(VariableDeclaration, self).__init__(name, variable_type, value, scope, source)
