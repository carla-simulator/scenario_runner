from srunner.osc2.symbol_manager.base_symbol import BaseSymbol
from srunner.osc2.symbol_manager.enum_symbol import EnumValueRefSymbol

# There are two types of inheritance
# 1. Unconditional inheritance
# 2. Conditional inheritance

# For method members, inheritance types can also override existing method implementations
# Scenarios and actions that belong to actors can only be inherited
# from scenarios or actions that belong to actors of the same type or more general type.
# and scenarios and actions that do not belong to actors can only be inherited from scenarios
# or actions that do not belong to roles.(action and actor)


class InheritsConditionSymbol(BaseSymbol):
    def __init__(self, name, scope):
        super().__init__(name, scope)

    def is_key_found(self, sym):
        if sym.name in self.symbols and sym.name:
            return True
        if self.enclosing_scope is not None:
            return self.enclosing_scope.is_key_found(sym)
        return False


class InheritSymbol(BaseSymbol):
    def __init__(self, name, scope, super_class_scope):
        super().__init__(name, scope)
        self.super_class_scope = super_class_scope

    # As long as resolve, get_number_of_symbols,
    # and get_child_symbol are overloaded at the same time, it's logically okay
    def resolve(self, name):
        s = self.symbols.get(name)
        if s is not None:
            return s
        if self.super_class_scope is not None:
            return self.super_class_scope.resolve(name)
        return None

    def get_parent_scope(self):
        if self.super_class_scope is None:
            return self.get_enclosing_scope()
        else:
            return self.super_class_scope

    def get_number_of_symbols(self):
        return self.super_class_scope.get_number_of_symbols()

    def get_child_symbol(self, i):
        return list(self.super_class_scope.symbols.values())[i]

    def __str__(self):
        buf = "inherits: " + self.name
        return buf


class StructInhertsSymbol(InheritSymbol):
    def __init__(self, name, scope, super_class_scope):
        super().__init__(name, scope, super_class_scope)


class ActorInhertsSymbol(InheritSymbol):
    def __init__(self, name, scope, super_class_scope):
        super().__init__(name, scope, super_class_scope)


class ActionInhertsSymbol(InheritSymbol):
    def __init__(self, QualifiedBehaviorSymbol, super_class_scope):
        super().__init__(
            QualifiedBehaviorSymbol.name,
            QualifiedBehaviorSymbol.scope,
            super_class_scope,
        )


class ScenarioInhertsSymbol(InheritSymbol):
    def __init__(self, QualifiedBehaviorSymbol, super_class_scope):
        super().__init__(
            QualifiedBehaviorSymbol.name,
            QualifiedBehaviorSymbol.scope,
            super_class_scope,
        )
