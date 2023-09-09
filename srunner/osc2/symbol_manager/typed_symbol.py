from srunner.osc2.symbol_manager.base_symbol import BaseSymbol

# This interface tags user-defined symbols that have static type information,
# like variables and functions.

# argumentListSpecification : argumentSpecification (',' argumentSpecification)*;
# argumentSpecification : argumentName ':' typeDeclarator ('=' defaultValue)?;
# argumentName : Identifier;

# typeDeclarator : nonAggregateTypeDeclarator | aggregateTypeDeclarator;
# nonAggregateTypeDeclarator : primitiveType | typeName | qualifiedBehaviorName;

# An aggregate type serves as a container for the unified members of another type.
# Currently, ASAM OpenSCENARIO only provides list types as aggregation types.
# aggregateTypeDeclarator : listTypeDeclarator;
# listTypeDeclarator : 'list' 'of' nonAggregateTypeDeclarator;
# primitiveType : 'int' | 'uint' | 'float' | 'bool' | 'string';
# typeName : Identifier;


# Here the ListMemberSymbol corresponds to the argumentSpecification:
# argumentName ':' typeDeclarator ('=' defaultValue)?;
class TypedSymbol(BaseSymbol):
    def __init__(self, name, scope, type, value=None):
        super().__init__(name, scope)
        self.type = type
        self.value = value

    def get_type(self):
        return self.type

    def set_type(self, type):
        self.type = type


class BoolSymbol(TypedSymbol):
    def __init__(self, scope, value=None):
        super().__init__("bool", scope, "bool", value)

    def __str__(self):
        buf = "bool"
        if self.value is not None:
            buf += " : "
            buf += str(self.value)
        return buf


class IntSymbol(TypedSymbol):
    def __init__(self, scope, type, value=None):
        super().__init__(type, scope, type, value)

    def __str__(self):
        buf = str(self.type)
        if self.value is not None:
            buf += ":"
            buf += str(self.value)
        return buf


class PhysicalSymbol(TypedSymbol):
    def __init__(self, scope, type, value=None):
        super().__init__("physical", scope, type, value)

    def __str__(self):
        buf = self.__class__.__name__
        buf += " : "
        buf += str(self.type)
        if self.value is not None:
            buf += ","
            buf += str(self.value)
        return buf


class IdentifierSymbol(TypedSymbol):
    def __init__(self, name, scope, value=None):
        super().__init__(name, scope, "identifier", value)

    def __str__(self):
        buf = "identifier"
        buf += " : "
        buf += self.name
        if self.value is not None:
            buf += " , "
            buf += str(self.value)
        return buf


class StringSymbol(TypedSymbol):
    def __init__(self, scope, value=None):
        super().__init__("string", scope, "string", value)

    def __str__(self):
        buf = "string"
        if self.value is not None:
            buf += " : "
            buf += str(self.value)
        return buf


class FloatSymbol(TypedSymbol):
    def __init__(self, scope, value=None):
        super().__init__("float", scope, "float", value)

    def __str__(self):
        buf = "float"
        if self.value is not None:
            buf += ":"
            buf += str(self.value)
        return buf
