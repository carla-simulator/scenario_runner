from srunner.osc2.symbol_manager.base_symbol import BaseSymbol
from srunner.osc2.symbol_manager.local_scope import LocalScope

###############################################################
# There are two types of modifiers
# 1. Atomic modifiers
# 2. Compound modifiers


class ModifierSymbol(BaseSymbol):
    def __init__(self, name, scope):
        super().__init__(name, scope)
        self.declaration_address = None


class ModifierInvocationSymbol(LocalScope):
    def __init__(self, behavior_name, scope, actor_name=None):
        name = ""
        if actor_name is not None:
            name += actor_name + "."
        name += behavior_name

        super().__init__(scope)
        self.actor_name = actor_name
        self.behavior_name = behavior_name
        self.name = name

    def __str__(self):
        buf = "ModifierInvocation"
        buf += " : "
        if self.actor_name is not None:
            buf += self.actor_name + "."
        buf += self.behavior_name
        return buf
