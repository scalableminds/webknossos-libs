"""
This package package provides methods to modify or interact with WEBKNOSSOS resources.

The core classes and can be created and saved as shown:

<div markdown="1" style="overflow-x: auto;">
<div markdown="1" style="width: max-content;">
|                  | Create ✨                                                                                                                                             | Open/Load 📂                                                                                                                                           | Save 💾                                                                       | Download 📥                                                                                       | Upload 📤                                                                         |
|------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| **`Dataset`**    | [`Dataset(…)`](webknossos/dataset/dataset.html#webknossos.dataset.dataset.Dataset),<br/>[`Dataset.from_images(…)`](webknossos/dataset/dataset.html#webknossos.dataset.dataset.Dataset.from_images) | [`Dataset.open(…)`](webknossos/dataset/dataset.html#webknossos.dataset.dataset.Dataset.open),<br/>[`Dataset.open_remote(…)`](webknossos/dataset/dataset.html#webknossos.dataset.dataset.Dataset.open_remote) | *N/A*[^ds-save]                                                               | [`Dataset.download(…)`](webknossos/dataset/dataset.html#webknossos.dataset.dataset.Dataset.download)                         | [`dataset.upload(…)`](webknossos/dataset/dataset.html#webknossos.dataset.dataset.Dataset.upload)              |
| **`Annotation`** | [`Annotation(…)`](webknossos/annotation/annotation.html#webknossos.dataset.dataset.Annotation)                                                                          | [`Annotation.load(…)`](webknossos/annotation/annotation.html#Annotation.load)                                                                          | [`annotation.save(…)`](webknossos/annotation/annotation.html#webknossos.dataset.dataset.Annotation.save) | [`Annotation.download(…)`](webknossos/annotation/annotation.html#Annotation.webknossos.dataset.dataset.download)[^anno-down] | [`annotation.upload(…)`](webknossos/annotation/annotation.html#webknossos.dataset.dataset.Annotation.upload)  |
| **`Skeleton`**   | [`Skeleton(…)`](webknossos/skeleton/skeleton.html#webknossos.dataset.dataset.webknossos.dataset.dataset.Skeleton)                                                                                  | [`Skeleton.load(…)`](webknossos/skeleton/skeleton.html#webknossos.dataset.dataset.Skeleton.load)                                                                                  | [`skeleton.save(…)`](webknossos/skeleton/skeleton.html#webknossos.dataset.dataset.Skeleton.save)         | *only in an annotation*                                                                           | *only in an annotation*                                                            |
</div>
</div>

[^ds-save]: Since the state of a dataset is continuously persisted it cannot be saved.
[^anno-down]: Downloaded annotations are not persisted on disk automatically.

Additionally, we provide the various geometrical primitives, e.g. `Vec3Int`, `BoundingBox` and `Mag`.

The `User`, `Project` and `Task` classes provide WEBKNOSSOS server interactions for administration purposes.
Server interactions may require [authentication](webknossos/client/context.html) e.g. via `webknossos_context`.
"""

# The table above contains zero-width spaces in the code examples after each dot to enforce correct line-breaks.
# ruff: noqa: F403
from .administration import *
from .annotation import *
from .client import *
from .dataset import *
from .geometry import *
from .skeleton import *
