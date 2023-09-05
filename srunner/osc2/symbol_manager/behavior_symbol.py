from srunner.osc2.symbol_manager.base_symbol import BaseSymbol
from srunner.osc2.symbol_manager.local_scope import LocalScope


class BehaviorInvocationSymbol(BaseSymbol):
    def __init__(self, behavior_name, scope, actor_name=None):
        name = ""
        if actor_name is not None:
            name += actor_name + "."
        name += behavior_name
        super().__init__(name, scope)
        self.actor_name = actor_name
        self.behavior_name = behavior_name

    def __str__(self):
        buf = self.__class__.__name__
        buf += " : "
        if self.actor_name is not None:
            buf += self.actor_name + "."
        buf += self.behavior_name
        return buf


class BehaviorWithScope(LocalScope):
    def __init__(self, scope):
        super().__init__(scope)
        self.name = "with"

    def __str__(self):
        return "with"
