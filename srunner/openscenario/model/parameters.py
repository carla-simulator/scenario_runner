"""Typed OpenSCENARIO values, parameters, variables, and evaluation state."""

from __future__ import absolute_import

import operator


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


class ParameterValue(object):
    """A declared OSC value with its lexical representation retained."""

    _CONVERTERS = {
        "string": lambda value: value,
        "boolean": lambda value: str(value).lower() == "true",
        "int": lambda value: int(value),
        "integer": lambda value: int(value),
        "unsignedInt": lambda value: _unsigned_int(value),
        "double": lambda value: float(value),
        "float": lambda value: float(value),
        "unsignedShort": lambda value: _unsigned_int(value),
        "unsignedByte": lambda value: _unsigned_int(value),
    }

    def __init__(self, value, value_type=None):
        self.raw = value
        self.value_type = value_type

    def convert(self, value_type=None):
        value_type = value_type or self.value_type
        converter = self._CONVERTERS.get(value_type, lambda value: value)
        if value_type == "boolean" and str(self.raw).lower() not in ("true", "false"):
            raise ValueError("Invalid boolean value {!r}".format(self.raw))
        return converter(self.raw)

    def __str__(self):
        return str(self.raw)


def _unsigned_int(value):
    result = int(value)
    if result < 0:
        raise ValueError("Unsigned value cannot be negative")
    return result


class ParameterStore(object):
    """Scoped, typed declaration store used by parameters and variables."""

    def __init__(self, declarations=None, values=None):
        self._declarations = {}
        self._values = {}
        for declaration in declarations or []:
            self.declare(declaration)
        for name, value in (values or {}).items():
            self.set(name, value)

    def declare(self, declaration):
        # Duplicate declarations are reported by SemanticValidator.  Keeping
        # construction non-throwing lets callers collect all document errors
        # before execution, including the legacy validation path.
        self._declarations[declaration.name] = declaration
        self._values[declaration.name] = ParameterValue(declaration.value, declaration.parameter_type)

    def names(self):
        return tuple(self._declarations)

    def declaration(self, name):
        if name not in self._declarations:
            raise KeyError("Unknown declaration '{}'".format(name))
        return self._declarations[name]

    def get(self, name, expected_type=None):
        name = _reference_name(name)
        if name not in self._values:
            raise KeyError("Unknown declaration '{}'".format(name))
        declaration = self._declarations[name]
        return self._values[name].convert(expected_type or declaration.parameter_type)

    def raw(self, name):
        name = _reference_name(name)
        if name not in self._values:
            raise KeyError("Unknown declaration '{}'".format(name))
        return self._values[name].raw

    def set(self, name, value):
        name = _reference_name(name)
        declaration = self.declaration(name)
        converted = ParameterValue(value, declaration.parameter_type).convert()
        self._values[name] = ParameterValue(converted, declaration.parameter_type)
        return converted

    def modify(self, name, rule, value):
        current = self.get(name)
        operand = self.resolve(value, self.declaration(name).parameter_type)
        if rule not in ("+", "*", "-", "multiply", "add", "subtract"):
            raise ValueError("Unsupported modification rule {!r}".format(rule))
        if rule in ("+", "add"):
            result = operator.add(current, operand)
        elif rule in ("-", "subtract"):
            result = operator.sub(current, operand)
        else:
            result = operator.mul(current, operand)
        return self.set(name, result)

    def resolve(self, expression, expected_type=None):
        if isinstance(expression, ParameterValue):
            return expression.convert(expected_type)
        text = str(expression)
        reference = _reference_name(text)
        if reference != text:
            return self.get(reference, expected_type)
        return ParameterValue(text, expected_type).convert(expected_type)


class VariableStore(ParameterStore):
    """Runtime variable namespace with explicit typed assignment semantics."""

    def modify(self, name, rule, value):
        aliases = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/"}
        rule = aliases.get(rule, rule)
        if rule not in ("+", "-", "*", "/"):
            raise ValueError("Unsupported variable modification rule {!r}".format(rule))
        current = self.get(name)
        operand = self.resolve(value, self.declaration(name).parameter_type)
        operations = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv}
        return self.set(name, operations[rule](current, operand))


class EvaluationContext(object):
    """Shared parameter/variable evaluation context for semantic execution."""

    def __init__(self, parameters=None, variables=None):
        self.parameters = parameters if parameters is not None else ParameterStore()
        self.variables = variables if variables is not None else VariableStore()

    def resolve(self, expression, expected_type=None):
        try:
            return self.variables.resolve(expression, expected_type)
        except KeyError:
            return self.parameters.resolve(expression, expected_type)


def _reference_name(value):
    text = str(value)
    return text[1:] if text.startswith("$") else text


class VariableDeclaration(ParameterDeclaration):
    """Placeholder semantic type for the distinct OSC variable namespace."""

    def __init__(self, name, variable_type, value, scope=ParameterScope.LOCAL, source=None):
        super(VariableDeclaration, self).__init__(name, variable_type, value, scope, source)
