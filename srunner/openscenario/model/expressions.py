"""Safe evaluation of OpenSCENARIO parameter references and ``${...}`` expressions.

Expressions are parsed with :mod:`ast` and interpreted by a whitelist-based
walker; nothing is ever passed to ``eval`` or ``exec``.  The supported language
is the one defined by ASAM OpenSCENARIO XML 1.1+ (section "Expressions"):
numeric and boolean literals, ``$name`` parameter references, the arithmetic
operators ``+ - * / %``, the Boolean operators ``not and or`` and the functions
``round floor ceil sqrt pow sin cos tan asin acos atan sign abs max min``.
"""

from __future__ import absolute_import

import ast
import math
import re

from srunner.openscenario.validation.diagnostics import Diagnostic, DiagnosticError

NAME_PATTERN = r"[A-Za-z_][A-Za-z0-9_]*"
INTEGER_TYPES = ("int", "integer", "unsignedInt", "unsignedShort", "unsignedByte")
DOUBLE_TYPES = ("double", "float")
NUMERIC_TYPES = INTEGER_TYPES + DOUBLE_TYPES
MAX_EXPRESSION_LENGTH = 1024

_REFERENCE = re.compile(r"\$(" + NAME_PATTERN + r")")
_FULL_REFERENCE = re.compile(r"^\$(" + NAME_PATTERN + r")$")
_PLACEHOLDER_PREFIX = "_osc_ref_"
_INT_LIMIT = 2 ** 63
_UNSIGNED_LIMITS = {"unsignedShort": 2 ** 16, "unsignedByte": 2 ** 8, "unsignedInt": 2 ** 32}


class EvaluationError(DiagnosticError):
    """An expression or reference could not be evaluated; carries a Diagnostic."""

    def __init__(self, message, code=Diagnostic.INVALID_EXPRESSION, expression=None, source=None):
        self.expression = expression
        if expression is not None:
            message = "{} (in {!r})".format(message, expression)
        self.diagnostic = Diagnostic("semantic_error", message, source=source, code=code)
        super(EvaluationError, self).__init__(str(self.diagnostic))

    @property
    def diagnostics(self):
        return [self.diagnostic]

    def __str__(self):
        return str(self.diagnostic)


class UnknownReferenceError(EvaluationError, KeyError):
    """A ``$name`` reference does not resolve.  Also a KeyError for lookup callers."""

    def __init__(self, name, expression=None, source=None):
        self.name = name
        super(UnknownReferenceError, self).__init__(
            "Unknown reference '{}'".format(name), Diagnostic.UNKNOWN_REFERENCE, expression, source)


def is_expression(text):
    """Return True if ``text`` uses the ``${...}`` expression notation."""
    if not isinstance(text, str):
        return False
    text = text.strip()
    return text.startswith("${") and text.endswith("}")


def reference_name(text):
    """Return the referenced name for a plain ``$name`` reference, else None."""
    if not isinstance(text, str):
        return None
    match = _FULL_REFERENCE.match(text.strip())
    return match.group(1) if match else None


def references(text):
    """Return the names referenced by a literal, ``$name`` or ``${...}`` text."""
    if is_expression(text):
        return list(_parse(text)[1].values())
    name = reference_name(text)
    return [name] if name is not None else []


def check_expression(text):
    """Validate the syntax and operator set of an expression without evaluating it."""
    tree, placeholders = _parse(text)
    _StaticChecker(text, placeholders).visit(tree.body)
    return sorted(set(placeholders.values()))


def evaluate(text, lookup):
    """Evaluate a ``${...}`` expression.  ``lookup(name)`` returns a typed parameter value."""
    tree, placeholders = _parse(text)
    return _Evaluator(text, placeholders, lookup).visit(tree.body)


def coerce(value, expected_type, expression=None):
    """Convert an expression result to ``expected_type`` using the OSC implicit conversion rules."""
    if not expected_type:
        return value
    if expected_type == "boolean":
        if isinstance(value, bool):
            return value
        raise EvaluationError("Arithmetic result {!r} where a boolean is expected".format(value),
                              Diagnostic.TYPE_MISMATCH, expression)
    if expected_type in INTEGER_TYPES:
        if isinstance(value, bool) or not isinstance(value, int):
            raise EvaluationError("Result {!r} cannot be implicitly converted to {}; use round, floor or ceil"
                                  .format(value, expected_type), Diagnostic.TYPE_MISMATCH, expression)
        return check_integer_range(value, expected_type, expression)
    if expected_type in DOUBLE_TYPES:
        if isinstance(value, bool):
            raise EvaluationError("Boolean result where {} is expected".format(expected_type),
                                  Diagnostic.TYPE_MISMATCH, expression)
        return float(value)
    if expected_type == "string":
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)
    raise EvaluationError("Expressions are not defined for type {}".format(expected_type),
                          Diagnostic.TYPE_MISMATCH, expression)


def check_integer_range(value, value_type, expression=None):
    if value_type in _UNSIGNED_LIMITS and not 0 <= value < _UNSIGNED_LIMITS[value_type]:
        raise EvaluationError("Value {} is out of range for {}".format(value, value_type),
                              Diagnostic.TYPE_MISMATCH, expression)
    if abs(value) >= _INT_LIMIT:
        raise EvaluationError("Integer overflow", Diagnostic.ARITHMETIC_ERROR, expression)
    return value


def _parse(text):
    if not is_expression(text):
        raise EvaluationError("Not an expression", expression=text)
    if len(text) > MAX_EXPRESSION_LENGTH:
        raise EvaluationError("Expression exceeds {} characters".format(MAX_EXPRESSION_LENGTH), expression=text)
    body = text.strip()[2:-1]
    if _PLACEHOLDER_PREFIX in _REFERENCE.sub("", body):
        raise EvaluationError("Invalid identifier in expression", expression=text)
    placeholders = {}

    def substitute(match):
        placeholder = _PLACEHOLDER_PREFIX + match.group(1)
        placeholders[placeholder] = match.group(1)
        return placeholder

    try:
        tree = ast.parse(_REFERENCE.sub(substitute, body).strip(), mode="eval")
    except (SyntaxError, ValueError, RecursionError, MemoryError) as error:
        raise EvaluationError("Invalid expression syntax", expression=text) from error
    return tree, placeholders


def _literal(node):
    if isinstance(node, ast.Constant):
        return node.value
    if type(node).__name__ == "Num":  # Python < 3.8
        return node.n
    return None


def _is_literal_number(node):
    value = _literal(node)
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class _StaticChecker(object):
    """Checks the expression structure only; shared base of the evaluator."""

    _UNARY = (ast.USub, ast.UAdd, ast.Not)
    _BINARY = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod)
    _BOOLEAN = (ast.And, ast.Or)
    FUNCTION_ARITY = {
        "round": 1, "floor": 1, "ceil": 1, "sqrt": 1, "sin": 1, "cos": 1, "tan": 1, "asin": 1,
        "acos": 1, "atan": 1, "sign": 1, "abs": 1, "max": 2, "min": 2, "pow": 2,
    }

    def __init__(self, text, placeholders):
        self.text = text
        self.placeholders = placeholders

    def error(self, message, code=Diagnostic.INVALID_EXPRESSION):
        return EvaluationError(message, code, self.text)

    def visit(self, node):
        try:
            if isinstance(node, (ast.Constant,)) or type(node).__name__ == "Num":
                return self.literal(node)
            if isinstance(node, ast.Name):
                return self.name(node)
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, self._UNARY):
                return self.unary(node)
            if isinstance(node, ast.BinOp) and isinstance(node.op, self._BINARY):
                return self.binary(node)
            if isinstance(node, ast.BoolOp) and isinstance(node.op, self._BOOLEAN):
                return self.boolean(node)
            if isinstance(node, ast.Call):
                return self.call(node)
        except RecursionError as error:
            raise self.error("Expression is nested too deeply") from error
        construct = type(getattr(node, "op", node)).__name__
        raise self.error("Unsupported expression construct '{}'".format(construct))

    def literal(self, node):
        if not _is_literal_number(node):
            raise self.error("Unsupported literal {!r}".format(_literal(node)))
        return _literal(node)

    def name(self, node):
        if node.id in self.placeholders:
            return None
        if node.id in ("true", "false"):
            return node.id == "true"
        raise self.error("Unknown identifier '{}'; parameter references need a '$' prefix".format(node.id))

    def unary(self, node):
        self.visit(node.operand)

    def binary(self, node):
        self.visit(node.left)
        self.visit(node.right)

    def boolean(self, node):
        for value in node.values:
            self.visit(value)

    def call(self, node):
        if not isinstance(node.func, ast.Name) or node.func.id not in self.FUNCTION_ARITY:
            raise self.error("Unsupported function")
        if node.keywords or len(node.args) != self.FUNCTION_ARITY[node.func.id]:
            raise self.error("Function '{}' expects {} argument(s)".format(
                node.func.id, self.FUNCTION_ARITY[node.func.id]))
        for argument in node.args:
            self.visit(argument)


class _Evaluator(_StaticChecker):

    def __init__(self, text, placeholders, lookup):
        super(_Evaluator, self).__init__(text, placeholders)
        self.lookup = lookup

    def _finite(self, value):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            raise self.error("Result is NaN or infinite", Diagnostic.ARITHMETIC_ERROR)
        if isinstance(value, int) and not isinstance(value, bool):
            check_integer_range(value, None, self.text)
        return value

    def _numeric(self, value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise self.error("Operator requires numeric operands, got {!r}".format(value),
                             Diagnostic.TYPE_MISMATCH)
        return value

    def _boolean(self, node):
        value = self.visit(node)
        if isinstance(value, bool):
            return value
        if _is_literal_number(node) and value in (0, 1):
            return bool(value)
        raise self.error("Boolean operator requires boolean operands, got {!r}".format(value),
                         Diagnostic.TYPE_MISMATCH)

    def name(self, node):
        if node.id not in self.placeholders:
            return super(_Evaluator, self).name(node)
        reference = self.placeholders[node.id]
        try:
            value = self.lookup(reference)
        except KeyError as error:
            raise UnknownReferenceError(reference, self.text) from error
        return self._typed_reference(reference, value)

    def _typed_reference(self, reference, value):
        if isinstance(value, (bool, int, float)):
            return value
        text = str(value).strip()
        if text in ("true", "false"):
            return text == "true"
        for converter in (int, float):
            try:
                return self._finite(converter(text))
            except ValueError:
                continue
        raise self.error("Parameter '{}' of non-numeric value {!r} cannot be used in an expression".format(
            reference, value), Diagnostic.TYPE_MISMATCH)

    def unary(self, node):
        if isinstance(node.op, ast.Not):
            return not self._boolean(node.operand)
        value = self._numeric(self.visit(node.operand))
        return -value if isinstance(node.op, ast.USub) else value

    def binary(self, node):
        left = self._numeric(self.visit(node.left))
        right = self._numeric(self.visit(node.right))
        try:
            if isinstance(node.op, ast.Add):
                return self._finite(left + right)
            if isinstance(node.op, ast.Sub):
                return self._finite(left - right)
            if isinstance(node.op, ast.Mult):
                return self._finite(left * right)
            if isinstance(node.op, ast.Div):
                return self._finite(float(left) / float(right))
            return self._finite(self._remainder(left, right))
        except ZeroDivisionError as error:
            raise self.error("Division by zero", Diagnostic.ARITHMETIC_ERROR) from error
        except OverflowError as error:
            raise self.error("Arithmetic overflow", Diagnostic.ARITHMETIC_ERROR) from error

    @staticmethod
    def _remainder(left, right):
        # OSC '%' is the truncating remainder (sign of the dividend), not Python's modulo.
        if isinstance(left, int) and isinstance(right, int):
            if right == 0:
                raise ZeroDivisionError()
            result = abs(left) % abs(right)
            return -result if left < 0 else result
        if right == 0:
            raise ZeroDivisionError()
        return math.fmod(left, right)

    def boolean(self, node):
        values = [self._boolean(value) for value in node.values]
        return all(values) if isinstance(node.op, ast.And) else any(values)

    def call(self, node):
        super(_Evaluator, self).call(node)
        function = node.func.id
        arguments = [self._numeric(self.visit(argument)) for argument in node.args]
        try:
            return self._finite(self._apply(function, arguments))
        except (ValueError, OverflowError) as error:
            if isinstance(error, EvaluationError):
                raise
            raise self.error("Arithmetic error in {}(): {}".format(function, error),
                             Diagnostic.ARITHMETIC_ERROR) from error

    @staticmethod
    def _apply(function, arguments):  # pylint: disable=too-many-return-statements
        value = arguments[0]
        if function == "round":
            return int(math.floor(abs(value) + 0.5)) * (1 if value >= 0 else -1)
        if function == "floor":
            return int(math.floor(value))
        if function == "ceil":
            return int(math.ceil(value))
        if function == "sign":
            return type(value)((value > 0) - (value < 0))
        if function == "abs":
            return abs(value)
        if function in ("max", "min"):
            result = max(arguments) if function == "max" else min(arguments)
            return float(result) if any(isinstance(item, float) for item in arguments) else result
        if function == "pow":
            return math.pow(float(arguments[0]), float(arguments[1]))
        return getattr(math, function)(float(value))
