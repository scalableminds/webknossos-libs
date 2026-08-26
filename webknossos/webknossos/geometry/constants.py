from typing import Final, Literal

C_AXIS: Final = "c"
X_AXIS: Final = "x"
Y_AXIS: Final = "y"
Z_AXIS: Final = "z"
T_AXIS: Final = "t"

TCXYZ_AXES: Final = (T_AXIS, C_AXIS, X_AXIS, Y_AXIS, Z_AXIS)
CXYZ_AXES: Final = (C_AXIS, X_AXIS, Y_AXIS, Z_AXIS)
TXYZ_AXES: Final = (T_AXIS, X_AXIS, Y_AXIS, Z_AXIS)
XYZ_AXES: Final = (X_AXIS, Y_AXIS, Z_AXIS)
XY_AXES: Final = (X_AXIS, Y_AXIS)

XYZAxis = Literal["x", "y", "z"]
