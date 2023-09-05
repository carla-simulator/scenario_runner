import copy

from srunner.osc2.symbol_manager.local_scope import LocalScope
from srunner.osc2.symbol_manager.symbol import Symbol
from srunner.osc2.utils.log_manager import *
from srunner.osc2.utils.tools import *


class BaseSymbol(Symbol):
    def __init__(self, name, scope):
        super().__init__(name, scope)
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
        if sym.name in self.symbols and sym.name:
            return True
        if self.enclosing_scope is not None:
            return self.enclosing_scope.is_key_found(sym)
        return False

    def define(self, sym, ctx):
        if issubclass(type(sym), LocalScope):
            self.symbols[sym.__class__.__name__ + str(ctx.line) + str(ctx.column)] = sym
        else:
            if is_multi_name(sym.name):
                names = multi_field_name_split(sym.name)
                for sub_sym_name in names:
                    sub_sym = copy.deepcopy(sym)
                    sub_sym.name = sub_sym_name
                    if self.is_key_found(sub_sym):
                        msg = sub_sym.name + " is already defined!"
                        LOG_ERROR(msg, ctx)
                    else:
                        self.symbols[sub_sym.name] = sub_sym
            else:
                if self.is_key_found(sym):
                    sep = "#"
                    name = sym.name.split(sep, 1)[0]
                    msg = name + " is already defined!"
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
        buf = self.__class__.__name__
        buf += " : "
        buf += self.name
        return buf
