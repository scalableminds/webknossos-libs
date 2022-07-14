"""
This package package provides methods to modify or interact with webKnossos resources.

The core classes and can be created and saved as shown:

<div markdown="1" style="overflow-x: auto;">
<div markdown="1" style="width: max-content;">
|                  | Create ✨                                                                    | Open/Load 📂                                                                                                                                           | Save 💾                                                                       | Download 📥                                                                                       | Upload 📤                                                                         |
|------------------|------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| **`Dataset`**    | [`Dataset(…)`](webknossos/dataset/dataset.html#Dataset.__init__)             | [`Dataset.open(…)`](webknossos/dataset/dataset.html#Dataset.open),<br/>[`Dataset.open_remote(…)`](webknossos/dataset/dataset.html#Dataset.open_remote) | *N/A*[^ds-save]                                                               | [`Dataset.download(…)`](webknossos/dataset/dataset.html#Dataset.download)                         | [`dataset.upload(…)`](webknossos/dataset/dataset.html#Dataset.upload)              |
| **`Annotation`** | [`Annotation(…)`](webknossos/annotation/annotation.html#Annotation.__init__) | [`Annotation.load(…)`](webknossos/annotation/annotation.html#Annotation.load)                                                                          | [`annotation.save(…)`](webknossos/annotation/annotation.html#Annotation.save) | [`Annotation.download(…)`](webknossos/annotation/annotation.html#Annotation.download)[^anno-down] | [`annotation.upload(…)`](webknossos/annotation/annotation.html#Annotation.upload)  |
| **`Skeleton`**   | [`Skeleton(…)`](webknossos/skeleton/skeleton.html#Skeleton.__init__)         | [`Skeleton.load(…)`](webknossos/skeleton/skeleton.html#Skeleton.load)                                                                                  | [`skeleton.save(…)`](webknossos/skeleton/skeleton.html#Skeleton.save)         | *only in an annotation*                                                                           | *only in an annotation*                                                            |
</div>
</div>

[^ds-save]: Since the state of a dataset is continously persisted it cannot be saved.
[^anno-down]: Downloaded annotations are not persisted on disk automatically.

Additionally, we provide the geometrical primitives `Vec3Int`, `BoundingBox` and `Mag`.

The `User`, `Project` and `Task` classes provide webKnossos server interactions for administration purposes.
Server interactions may require [authentication](webknossos/client/context.html) e.g. via `webknossos_context`.
"""

# The table above contains zero-width spaces in the code examples after each dot to enforce correct line-breaks.

from webknossos.administration import *
from webknossos.annotation import *
from webknossos.client import *
from webknossos.dataset import *
from webknossos.geometry import *
from webknossos.skeleton import *
