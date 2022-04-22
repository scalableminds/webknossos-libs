from enum import Enum, unique
from typing import Union


@unique
class SamplingModes(Enum):
    ANISOTROPIC = "anisotropic"
    ISOTROPIC = "isotropic"
    CONSTANT_Z = "constant_z"

    @classmethod
    def parse(cls, mode: Union[str, "SamplingModes"]) -> "SamplingModes":
        if mode == "auto":
            return cls.ANISOTROPIC
        else:
            return cls(mode)
