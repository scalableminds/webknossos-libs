from enum import Enum


class LengthUnit(Enum):
    YOCTOMETER = "yoctometer"
    ZEPTOMETER = "zeptometer"
    ATTOMETER = "attometer"
    FEMTOMETER = "femtometer"
    PICOMETER = "picometer"
    NANOMETER = "nanometer"
    MICROMETER = "micrometer"
    MILLIMETER = "millimeter"
    CENTIMETER = "centimeter"
    DECIMETER = "decimeter"
    METER = "meter"
    HECTOMETER = "hectometer"
    KILOMETER = "kilometer"
    MEGAMETER = "megameter"
    GIGAMETER = "gigameter"
    TERAMETER = "terameter"
    PETAMETER = "petameter"
    EXAMETER = "exameter"
    ZETTAMETER = "zettameter"
    YOTTAMETER = "yottameter"
    ANGSTROM = "angstrom"
    INCH = "inch"
    FOOT = "foot"
    YARD = "yard"
    MILE = "mile"
    PARSEC = "parsec"


_STR_TO_UNIT_MAP = {
    "ym": LengthUnit.YOCTOMETER,
    "yoctometer": LengthUnit.YOCTOMETER,
    "zm": LengthUnit.ZEPTOMETER,
    "zeptometer": LengthUnit.ZEPTOMETER,
    "am": LengthUnit.ATTOMETER,
    "attometer": LengthUnit.ATTOMETER,
    "fm": LengthUnit.FEMTOMETER,
    "femtometer": LengthUnit.FEMTOMETER,
    "pm": LengthUnit.PICOMETER,
    "picometer": LengthUnit.PICOMETER,
    "nm": LengthUnit.NANOMETER,
    "nanometer": LengthUnit.NANOMETER,
    "µm": LengthUnit.MICROMETER,
    "micrometer": LengthUnit.MICROMETER,
    "mm": LengthUnit.MILLIMETER,
    "millimeter": LengthUnit.MILLIMETER,
    "cm": LengthUnit.CENTIMETER,
    "centimeter": LengthUnit.CENTIMETER,
    "dm": LengthUnit.DECIMETER,
    "decimeter": LengthUnit.DECIMETER,
    "m": LengthUnit.METER,
    "meter": LengthUnit.METER,
    "hm": LengthUnit.HECTOMETER,
    "hectometer": LengthUnit.HECTOMETER,
    "km": LengthUnit.KILOMETER,
    "kilometer": LengthUnit.KILOMETER,
    "Mm": LengthUnit.MEGAMETER,
    "megameter": LengthUnit.MEGAMETER,
    "Gm": LengthUnit.GIGAMETER,
    "gigameter": LengthUnit.GIGAMETER,
    "Tm": LengthUnit.TERAMETER,
    "terameter": LengthUnit.TERAMETER,
    "Pm": LengthUnit.PETAMETER,
    "petameter": LengthUnit.PETAMETER,
    "Em": LengthUnit.EXAMETER,
    "exameter": LengthUnit.EXAMETER,
    "Zm": LengthUnit.ZETTAMETER,
    "zettameter": LengthUnit.ZETTAMETER,
    "Ym": LengthUnit.YOTTAMETER,
    "yottameter": LengthUnit.YOTTAMETER,
    "Å": LengthUnit.ANGSTROM,
    "angstrom": LengthUnit.ANGSTROM,
    "in": LengthUnit.INCH,
    "inch": LengthUnit.INCH,
    "ft": LengthUnit.FOOT,
    "foot": LengthUnit.FOOT,
    "yd": LengthUnit.YARD,
    "yard": LengthUnit.YARD,
    "mi": LengthUnit.MILE,
    "mile": LengthUnit.MILE,
    "pc": LengthUnit.PARSEC,
    "parsec": LengthUnit.PARSEC,
}

_LENGTH_UNIT_TO_NANOMETER = {
    LengthUnit.YOCTOMETER: 1e-15,
    LengthUnit.ZEPTOMETER: 1e-12,
    LengthUnit.ATTOMETER: 1e-9,
    LengthUnit.FEMTOMETER: 1e-6,
    LengthUnit.PICOMETER: 1e-3,
    LengthUnit.NANOMETER: 1.0,
    LengthUnit.MICROMETER: 1e3,
    LengthUnit.MILLIMETER: 1e6,
    LengthUnit.CENTIMETER: 1e7,
    LengthUnit.DECIMETER: 1e8,
    LengthUnit.METER: 1e9,
    LengthUnit.HECTOMETER: 1e11,
    LengthUnit.KILOMETER: 1e12,
    LengthUnit.MEGAMETER: 1e15,
    LengthUnit.GIGAMETER: 1e18,
    LengthUnit.TERAMETER: 1e21,
    LengthUnit.PETAMETER: 1e24,
    LengthUnit.EXAMETER: 1e27,
    LengthUnit.ZETTAMETER: 1e30,
    LengthUnit.YOTTAMETER: 1e33,
    LengthUnit.ANGSTROM: 0.1,
    LengthUnit.INCH: 25400000.0,
    LengthUnit.FOOT: 304800000.0,
    LengthUnit.YARD: 914400000.0,
    LengthUnit.MILE: 1609344000000.0,
    LengthUnit.PARSEC: 3.085677581e25,
}


def length_unit_from_str(unit: str) -> LengthUnit:
    if unit in _STR_TO_UNIT_MAP:
        return _STR_TO_UNIT_MAP[unit]
    raise ValueError(f"Unknown unit: {unit}")
