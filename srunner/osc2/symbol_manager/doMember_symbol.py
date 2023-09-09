from srunner.osc2.symbol_manager.base_symbol import BaseSymbol
from srunner.osc2.symbol_manager.behavior_symbol import BehaviorInvocationSymbol
from srunner.osc2.symbol_manager.wait_symbol import WaitSymbol


# domember scope
class DoMemberSymbol(BaseSymbol):
    def __init__(self, name, scope, op):
        super().__init__(name, scope)
        self.op = op

    def get_scope_name(self):
        return self.name

    def __str__(self):
        buf = self.__class__.__name__
        buf += " : "
        if self.name is not None:
            buf += self.name + ","
        buf += str(self.op)
        return buf
