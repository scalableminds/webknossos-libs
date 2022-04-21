from warnings import warn

from ._internal.utils import *  # pylint: disable=unused-wildcard-import,wildcard-import

warn(
    "[DEPRECATION] Using `wkcuber.utils` is deprecated. "
    "Please use the high-level APIs of the `webknossos` package instead.",
    DeprecationWarning,
)
