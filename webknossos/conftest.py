"""Root conftest, loaded before any test module imports webknossos.

The defaults in `webknossos.dataset.defaults` are read from the environment at
import time, so the test-wide overrides have to be set here.
"""

import os

# The shipped default shard shape is 1024**3. Tests only write a few kilobytes,
# but compressing or rewriting a shard always touches all of it, which dominates
# the runtime of the dataset tests. 256**3 keeps the same assertions at a
# fraction of the cost. `setdefault` keeps it overridable from the outside.
os.environ.setdefault("WK_DEFAULT_CHUNKS_PER_SHARD", "8,8,8")
