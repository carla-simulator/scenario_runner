from srunner.osc2.symbol_manager.base_scope import BaseScope
from srunner.osc2.utils.log_manager import *


class LocalScope(BaseScope):
    def __init__(self, scope):
        super().__init__(scope)

    # For local scopes, only internal naming conflicts are found
    def is_key_found(self, sym):
        if sym.name and sym.name in self.symbols:
            return True
        else:
            return False

    def define(self, sym, ctx):
        if issubclass(type(sym), LocalScope):
            self.symbols[sym.__class__.__name__ + str(ctx.line) + str(ctx.column)] = sym
        else:
            if self.is_key_found(sym):
                msg = sym.name + " is already defined!"
                LOG_ERROR(msg, ctx)
            else:
                self.symbols[sym.name] = sym

    def get_scope_name(self):
        return "local"
