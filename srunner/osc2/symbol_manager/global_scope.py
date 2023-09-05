from srunner.osc2.symbol_manager.base_scope import BaseScope


# Global scope
class GlobalScope(BaseScope):
    def __init__(self, scope):
        super().__init__(scope)

    def get_scope_name(self):
        return "global"

    def __str__(self):
        return "global"
