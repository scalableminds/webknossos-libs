import re
from math import log2
from functools import total_ordering


@total_ordering
class Mag:
    def __init__(self, mag):
        self.mag = None

        if isinstance(mag, int):
            self.mag = [mag] * 3
        elif isinstance(mag, list):
            self.mag = mag
        elif isinstance(mag, str):
            if re.match(r"^\d+$", mag) is not None:
                self.mag = [int(mag)] * 3
            elif re.match(r"^\d+-\d+-\d+$", mag) is not None:
                potential_mag = [int(m) for m in mag.split("-")]
                if len(potential_mag) == 3:
                    self.mag = potential_mag

        if self.mag is None:
            raise ValueError(
                "magnification must be int or a vector3 of ints or a string shaped like e.g. 2-2-1"
            )

        for m in self.mag:
            assert log2(m) % 1 == 0, "magnification needs to be power of 2."

    def __lt__(self, other):
        return max(self.mag) < (max(other.to_array()))

    def __eq__(self, other):
        is_equal = True
        for m1, m2 in zip(self.mag, other.mag):
            if m1 != m2:
                is_equal = False
        return is_equal

    def __str__(self):
        return self.to_layer_name()

    def to_layer_name(self):
        x, y, z = self.mag
        if x == y and y == z:
            return str(x)
        else:
            return "{}-{}-{}".format(x, y, z)

    def to_array(self):
        return self.mag

    def scaled_by(self, factor):
        return Mag([mag * factor for mag in self.mag])

    def scale_by(self, factor):
        self.mag = [mag * factor for mag in self.mag]

    def divide_by(self, d):
        self.mag = [mag // d for mag in self.mag]

    def divided_by(self, d):
        return Mag([mag // d for mag in self.mag])
