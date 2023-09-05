class PhysicalObject(object):
    #  All locations have been converted to meters and speeds to meters/second.
    def __init__(self, name: str, si_base_exponent: dict) -> None:
        self.physical_name = name
        self.si_base_exponent = si_base_exponent

    def __add__(self, right):
        physical_name = self.physical_name + "+" + right.physical_name
        si_base_exponent = self.si_base_exponent
        return PhysicalObject(physical_name, si_base_exponent)

    def __sub__(self, right):
        physical_name = self.physical_name + "-" + right.physical_name
        si_base_exponent = self.si_base_exponent
        return PhysicalObject(physical_name, si_base_exponent)

    def __truediv__(self, right):
        physical_name = self.physical_name + "/" + right.physical_name
        si_base_exponent = self.si_base_exponent
        for key, value in right.si_base_exponent.items():
            if key in si_base_exponent:
                si_base_exponent[key] -= value
            else:
                si_base_exponent[key] = -value
        return PhysicalObject(physical_name, si_base_exponent)

    # Multiplication
    def __mul__(self, right):
        physical_name = self.physical_name + "*" + right.physical_name
        si_base_exponent = self.si_base_exponent
        for key, value in right.si_base_exponent.items():
            if key in si_base_exponent:
                si_base_exponent[key] += value
            else:
                si_base_exponent[key] = value
        return PhysicalObject(physical_name, si_base_exponent)


class UnitObject(object):
    def __init__(self, name, physical, factor=1.0, offset=0) -> None:
        self.unit_name = name
        self.physical = physical
        self.factor = factor
        self.offset = offset

    def __str__(self) -> str:
        return str(self.unit_name)
