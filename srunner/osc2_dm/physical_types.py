import math
import random
import re
import sys
from typing import List

from srunner.osc2_dm.physical_object import *


class Range(object):
    def __init__(self, start, end):
        super().__init__()
        # Store everything using float
        self.start = start
        self.end = end

    # String initialization，[4..6]
    @classmethod
    def from_str(cls, s: str):
        values = s[1:-1].split("..")

        start = float(values[0]) if values[0] else 0
        end = float(values[1]) if values[1] else math.inf

        return cls(start, end)

    def __str__(self) -> str:
        return "[" + str(self.start) + ".." + str(self.end) + "]"

    def __neg__(self):
        self.start = -self.start
        self.end = -self.end
        return self

    def is_in_range(self, num) -> bool:
        if num >= self.start and num < self.end:
            return True
        else:
            return False

    def gen_single_value(self):
        return random.uniform(self.start, self.end)


class Physical(object):
    def __init__(self, num, unit):
        super().__init__()
        # num can be of the range type or the float type in the number category
        self.num = num
        self.unit = unit

    # from string init
    @classmethod
    def from_str(cls, s: str):
        # First match if it is a range type ([3..6] format), and then match pure number types
        if not s:
            return None
        match_obj = re.match(r"\[.*\]", s)
        if match_obj:
            range_num_indexs = match_obj.span()
            num_start = range_num_indexs[0]
            num_end = range_num_indexs[1]
            num = Range.from_str(s[num_start:num_end])
            unit = s[num_end:]
            return cls(num, unit)
        else:
            match_obj = re.match(r"\d+\.?\d*", s)
            if match_obj:
                nums_indexs = match_obj.span()
                num_start = nums_indexs[0]
                num_end = nums_indexs[1]
                num = float(s[num_start:num_end])
                unit = s[num_end:]
                return cls(num, unit)
            else:
                print("wrong physical")

    def is_in_range(self, value) -> bool:
        if isinstance(self.num, Range):
            return self.num.is_in_range(value)
        else:
            return value == self.num

    def is_single_value(self) -> bool:
        if isinstance(self.num, Range):
            return False
        else:
            return True

    def gen_single_value(self):
        if self.is_single_value():
            return self.num
        else:
            return self.num.gen_single_value()

    def gen_physical_value(self):
        # The speed unit in Carla is m/s, so it is automatically converted to m/s
        value = self.gen_single_value()
        value *= self.unit.factor
        value += self.unit.offset
        return value

    def __str__(self) -> str:
        return str(self.num) + str(self.unit.unit_name)

    # Addition
    def __add__(self, right):
        num = (
            self.num
            + ((right.num * right.unit.factor + right.unit.offset) - self.unit.offset)
            / self.unit.factor
        )
        factor = self.unit.factor
        offset = self.unit.offset + right.unit.offset / self.unit.factor
        name = self.unit.unit_name + "+" + right.unit.unit_name
        physical = self.unit.physical + right.unit.physical
        unit = UnitObject(name, physical, factor, offset)
        return Physical(num, unit)

    # Subtraction
    def __sub__(self, right):
        num = (
            self.num
            - ((right.num * right.unit.factor + right.unit.offset) - self.unit.offset)
            / self.unit.factor
        )
        factor = self.unit.factor
        offset = self.unit.offset + right.unit.offset / self.unit.factor
        name = self.unit.unit_name + "-" + right.unit.unit_name
        physical = self.unit.physical - right.unit.physical
        unit = UnitObject(name, physical, factor, offset)
        return Physical(num, unit)

    # Division
    def __truediv__(self, right):
        num = self.num / right.num
        factor = self.unit.factor / right.unit.factor
        offset = (
            self.unit.offset / right.unit.factor + right.unit.offset / self.unit.factor
        )
        name = self.unit.unit_name + "/" + right.unit.unit_name
        physical = self.unit.physical / right.unit.physical
        unit = UnitObject(name, physical, factor, offset)
        return Physical(num, unit)

    # Multiplication
    def __mul__(self, right):
        num = self.num * right.num
        factor = self.unit.factor * right.unit.factor
        offset = (
            self.unit.offset * right.unit.factor + right.unit.offset * self.unit.factor
        )
        name = self.unit.unit_name + "*" + right.unit.unit_name
        physical = self.unit.physical * right.unit.physical
        unit = UnitObject(name, physical, factor, offset)
        return Physical(num, unit)
