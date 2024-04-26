from srunner.osc2.symbol_manager.base_symbol import BaseSymbol
from srunner.osc2.symbol_manager.inherits_condition_symbol import StructInhertsSymbol


class StructSymbol(BaseSymbol):
    def __init__(self, name, scope):
        super().__init__(name, scope)

    def is_key_found(self, sym):
        if isinstance(sym, StructInhertsSymbol):
            # Do not repeat checks on inheritance and stop recursion
            return False
        if sym.name in self.symbols and sym.name:
            return True
        if self.enclosing_scope is not None:
            return self.enclosing_scope.is_key_found(sym)
        return False
