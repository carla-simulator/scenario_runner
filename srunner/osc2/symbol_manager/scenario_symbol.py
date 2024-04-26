from srunner.osc2.symbol_manager.inherits_condition_symbol import ScenarioInhertsSymbol
from srunner.osc2.symbol_manager.qualifiedBehavior_symbol import QualifiedBehaviorSymbol
from srunner.osc2.symbol_manager.scope import Scope
from srunner.osc2.utils.log_manager import *


class ScenarioSymbol(QualifiedBehaviorSymbol, Scope):
    def __init__(self, QualifiedBehaviorSymbol):
        super().__init__(QualifiedBehaviorSymbol.name, QualifiedBehaviorSymbol.scope)
        self.declaration_address = None

    def is_key_found(self, sym):
        if isinstance(sym, ScenarioInhertsSymbol):
            return False
        if sym.name in self.symbols and sym.name:
            return True
        if self.enclosing_scope is not None:
            return self.enclosing_scope.is_key_found(sym)
        return False

    def __str__(self):
        buf = self.__class__.__name__
        buf += " : "
        buf += self.name
        return buf
