from enum import Enum, unique


@unique
class SamplingModes(Enum):
    ANISOTROPIC = "anisotropic"
    ISOTROPIC = "isotropic"
    CONSTANT_Z = "constant_z"

    @classmethod
    def parse(cls, mode: str | "SamplingModes") -> "SamplingModes":
        if mode == "auto":
            return cls.ANISOTROPIC
        else:
            return cls(mode)
