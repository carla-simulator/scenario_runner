from srunner.osc2.symbol_manager.scope import Scope
from srunner.osc2.symbol_manager.symbol import Symbol
from srunner.osc2.utils.log_manager import *

# All symbols defined in this scope; This can include classes, functions, variables,
# or any other symbolic impl. It does not include things that are not based on symbols


class BaseScope(Scope):
    def __init__(self, scope: Scope):
        self.enclosing_scope = scope
        self.symbols = {}

    def resolve(self, name):
        s = self.symbols.get(name)
        if s is not None:
            return s
        if self.enclosing_scope is not None:
            return self.enclosing_scope.resolve(name)
        return None

    def is_key_found(self, sym):
        if sym.name and sym.name in self.symbols:
            return True
        if self.enclosing_scope is not None:
            return self.enclosing_scope.is_key_found(sym)
        return False

    def define(self, sym, ctx):
        if self.is_key_found(sym):
            msg = sym.name + " is already defined!"
            LOG_ERROR(msg, ctx)
        else:
            self.symbols[sym.name] = sym

    def get_enclosing_scope(self):
        return self.enclosing_scope

    def get_number_of_symbols(self):
        return len(self.symbols)

    def get_child_symbol(self, i):
        return list(self.symbols.values())[i]

    def __str__(self):
        buf = self.get_scope_name() + " : " + list(self.symbols.keys()).__str__()
        return buf
