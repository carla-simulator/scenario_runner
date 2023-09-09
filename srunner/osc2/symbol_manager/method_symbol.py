from srunner.osc2.symbol_manager.argument_symbol import ArgumentSpecificationSymbol
from srunner.osc2.symbol_manager.base_symbol import BaseSymbol


class MethodSymbol(BaseSymbol):
    def __init__(self, name, scope):
        super().__init__(name, scope)
        self.declaration_address = None

    # Repeated definitions are allowed for input parameters
    def is_key_found(self, sym):
        if isinstance(sym, ArgumentSpecificationSymbol):
            return False
        if sym.name in self.symbols and sym.name:
            return True
        if self.enclosing_scope is not None:
            return self.enclosing_scope.is_key_found(sym)
        return False
