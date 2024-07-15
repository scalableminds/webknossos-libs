from enum import Enum


class Unit(Enum):
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
    "ym": Unit.YOCTOMETER,
    "yoctometer": Unit.YOCTOMETER,
    "zm": Unit.ZEPTOMETER,
    "zeptometer": Unit.ZEPTOMETER,
    "am": Unit.ATTOMETER,
    "attometer": Unit.ATTOMETER,
    "fm": Unit.FEMTOMETER,
    "femtometer": Unit.FEMTOMETER,
    "pm": Unit.PICOMETER,
    "picometer": Unit.PICOMETER,
    "nm": Unit.NANOMETER,
    "nanometer": Unit.NANOMETER,
    "µm": Unit.MICROMETER,
    "micrometer": Unit.MICROMETER,
    "mm": Unit.MILLIMETER,
    "millimeter": Unit.MILLIMETER,
    "cm": Unit.CENTIMETER,
    "centimeter": Unit.CENTIMETER,
    "dm": Unit.DECIMETER,
    "decimeter": Unit.DECIMETER,
    "m": Unit.METER,
    "meter": Unit.METER,
    "hm": Unit.HECTOMETER,
    "hectometer": Unit.HECTOMETER,
    "km": Unit.KILOMETER,
    "kilometer": Unit.KILOMETER,
    "Mm": Unit.MEGAMETER,
    "megameter": Unit.MEGAMETER,
    "Gm": Unit.GIGAMETER,
    "gigameter": Unit.GIGAMETER,
    "Tm": Unit.TERAMETER,
    "terameter": Unit.TERAMETER,
    "Pm": Unit.PETAMETER,
    "petameter": Unit.PETAMETER,
    "Em": Unit.EXAMETER,
    "exameter": Unit.EXAMETER,
    "Zm": Unit.ZETTAMETER,
    "zettameter": Unit.ZETTAMETER,
    "Ym": Unit.YOTTAMETER,
    "yottameter": Unit.YOTTAMETER,
    "Å": Unit.ANGSTROM,
    "angstrom": Unit.ANGSTROM,
    "in": Unit.INCH,
    "inch": Unit.INCH,
    "ft": Unit.FOOT,
    "foot": Unit.FOOT,
    "yd": Unit.YARD,
    "yard": Unit.YARD,
    "mi": Unit.MILE,
    "mile": Unit.MILE,
    "pc": Unit.PARSEC,
    "parsec": Unit.PARSEC,
}

_UNIT_TO_CONVERSION_FACTOR = {
    Unit.YOCTOMETER: 1e-15,
    Unit.ZEPTOMETER: 1e-12,
    Unit.ATTOMETER: 1e-9,
    Unit.FEMTOMETER: 1e-6,
    Unit.PICOMETER: 1e-3,
    Unit.NANOMETER: 1.0,
    Unit.MICROMETER: 1e3,
    Unit.MILLIMETER: 1e6,
    Unit.CENTIMETER: 1e7,
    Unit.DECIMETER: 1e8,
    Unit.METER: 1e9,
    Unit.HECTOMETER: 1e11,
    Unit.KILOMETER: 1e12,
    Unit.MEGAMETER: 1e15,
    Unit.GIGAMETER: 1e18,
    Unit.TERAMETER: 1e21,
    Unit.PETAMETER: 1e24,
    Unit.EXAMETER: 1e27,
    Unit.ZETTAMETER: 1e30,
    Unit.YOTTAMETER: 1e33,
    Unit.ANGSTROM: 0.1,
    Unit.INCH: 25400000.0,
    Unit.FOOT: 304800000.0,
    Unit.YARD: 914400000.0,
    Unit.MILE: 1609344000000.0,
    Unit.PARSEC: 3.085677581e25,
}


def get_unit_from_str(unit: str) -> Unit:
    if unit in _STR_TO_UNIT_MAP:
        return _STR_TO_UNIT_MAP[unit]
    raise ValueError(f"Unknown unit: {unit}")


def get_conversion_factor(unit: Unit) -> float:
    return _UNIT_TO_CONVERSION_FACTOR[unit]
