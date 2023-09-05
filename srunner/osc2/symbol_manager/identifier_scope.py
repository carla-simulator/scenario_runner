from srunner.osc2.symbol_manager.local_scope import LocalScope


class IdentifierScope(LocalScope):
    def __init__(self, name, scope):
        super().__init__(scope)
        self.name = name

    def __str__(self):
        buf = self.name
        return buf
