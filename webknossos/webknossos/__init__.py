"""
This package package provides methods to modify or interact with WEBKNOSSOS resources.

The core classes and can be created and saved as shown:

<div markdown="1" style="overflow-x: auto;">
<div markdown="1" style="width: max-content;">
|                  | Create ✨                                                                                                                                             | Open/Load 📂                                                                                                                                           | Save 💾                                                                       | Download 📥                                                                                       | Upload 📤                                                                         |
|------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| **`Dataset`**    | [`Dataset(…)`](webknossos/dataset/dataset.html#webknossos.dataset.Dataset),<br/>[`Dataset.from_images(…)`](webknossos/dataset/dataset.html#webknossos.dataset.Dataset.from_images) | [`Dataset.open(…)`](webknossos/dataset/dataset.html#webknossos.dataset.Dataset.open),<br/>[`RemoteDataset.open(…)`](webknossos/dataset/remote_dataset.html#webknossos.dataset.RemoteDataset.open) | *N/A*[^ds-save]                                                               | [`RemoteDataset.download(…)`](webknossos/dataset/remote_dataset.html#webknossos.dataset.RemoteDataset.download)                         | [`dataset.upload(…)`](webknossos/dataset/dataset.html#webknossos.dataset.Dataset.upload)              |
| **`Annotation`** | [`Annotation(…)`](webknossos/annotation/annotation.html#webknossos.annotation.Annotation)                                                                          | [`Annotation.load(…)`](webknossos/annotation/annotation.html#webknossos.annotation.Annotation.load)                                                                          | [`annotation.save(…)`](webknossos/annotation/annotation.html#webknossos.annotation.Annotation.save) | [`Annotation.download(…)`](webknossos/annotation/annotation.html#webknossos.annotation.Annotation.download)[^anno-down] | [`annotation.upload(…)`](webknossos/annotation/annotation.html#webknossos.annotation.Annotation.upload)  |
</div>
</div>

[^ds-save]: Since the state of a dataset is continuously persisted it cannot be saved.
[^anno-down]: Downloaded annotations are not persisted on disk automatically.

Additionally, we provide the various geometrical primitives, e.g. `Vec3Int`, `BoundingBox` and `Mag`.

The `User`, `Team`, `Project` and `Task` classes provide WEBKNOSSOS server interactions for administration purposes.
Server interactions may require [authentication](webknossos/client/context.html) e.g. via `login(token="...")`.
"""

# The table above contains zero-width spaces in the code examples after each dot to enforce correct line-breaks.
# ruff: noqa: F403
from .administration import *
from .ai_model import *
from .annotation import *
from .client import *
from .dataset import *
from .dataset_properties import *
from .datastore import *
from .geometry import *
from .skeleton import *
from .utils import check_version_in_background
from .version import __version__ as current_version

if not current_version == "0.0.0":
    import multiprocessing
    import os

    if (
        multiprocessing.parent_process() is None
        and os.getenv("WEBKNOSSOS_SKIP_VERSION_CHECK", "False") != "True"
    ):
        os.environ["WEBKNOSSOS_SKIP_VERSION_CHECK"] = "True"
        # Schedule the version check to run non-blocking in a background thread
        check_version_in_background(current_version)
import os

# This needs to be set to make sure the encoding is not chunked when uploading
os.environ["AWS_REQUEST_CHECKSUM_CALCULATION"] = "WHEN_REQUIRED"
