"""Typed foundations for OpenSCENARIO parameters and future variables."""

from __future__ import absolute_import


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


class VariableDeclaration(ParameterDeclaration):
    """Placeholder semantic type for the distinct OSC variable namespace."""

    def __init__(self, name, variable_type, value, scope=ParameterScope.LOCAL, source=None):
        super(VariableDeclaration, self).__init__(name, variable_type, value, scope, source)
