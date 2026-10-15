"""Structured diagnostics shared by validation and execution boundaries."""

from __future__ import absolute_import


class Diagnostic(object):
    CATEGORIES = frozenset((
        "schema_error", "semantic_error", "unsupported_feature",
        "backend_capability_error", "approximation_warning", "runtime_error"))

    def __init__(self, category, message, path=None, severity="error", source=None):
        if category not in self.CATEGORIES:
            raise ValueError("Unknown diagnostic category: {}".format(category))
        self.category = category
        self.message = message
        self.path = path
        self.severity = severity
        self.source = source

    def __str__(self):
        location = " at {}".format(self.path) if self.path else ""
        return "[{}]{} {}".format(self.category, location, self.message)

    def __repr__(self):
        return "Diagnostic(category={!r}, message={!r}, path={!r})".format(
            self.category, self.message, self.path)


class ValidationError(ValueError):
    def __init__(self, diagnostics):
        self.diagnostics = list(diagnostics)
        super(ValidationError, self).__init__("; ".join(str(item) for item in self.diagnostics))
