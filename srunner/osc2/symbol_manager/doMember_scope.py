from srunner.osc2.symbol_manager.local_scope import LocalScope


# domember scope
class DoMemberScope(LocalScope):
    def __init__(self, name, scope):
        super().__init__(scope)
        self.name = name
