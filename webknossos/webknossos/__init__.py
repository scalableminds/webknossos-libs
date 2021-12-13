"""
This package package provides methods to modify or interact with webKnossos resources.

The core classes and can be created and saved as shown:

|                                                                    | Create ✨                                                            | Open/Load 📂                                                       | Save 💾                  | Download 📥                                                              | Upload 📤                                                                |
|--------------------------------------------------------------------|----------------------------------------------------------------------|--------------------------------------------------------------------|--------------------------|--------------------------------------------------------------------------|---------------------------------------------------------------------------|
| **[Dataset](webknossos/dataset/dataset.html#Dataset)**             | [`Dataset(…)`](webknossos/dataset/dataset.html#Dataset.__init__)     | [`Dataset.open(…)`](webknossos/dataset/dataset.html#Dataset.open)  | *N/A[^ds-save]*          | [`Dataset.download(…)`](webknossos/dataset/dataset.html#Dataset.download)| [`dataset_obj.upload(…)`](webknossos/dataset/dataset.html#Dataset.upload) |
| **[Annotation](webknossos/annotation/annotation.html#Annotation)** | *N/A yet*                                                            | `Annotation.load(…)`                                               | `annotation_obj.save(…)` | `Annotation.download`                                                    | `annotation_obj.upload(…)`                                                |
| **[Skeleton](webknossos/skeleton/skeleton.html#Skeleton)**         | [`Skeleton(…)`](webknossos/skeleton/skeleton.html#Skeleton.__init__) | `Skeleton.load(…)`                                                 | `skeleton_obj.save(…)`   | *only in an annotation*                                                  | *only in an annotation*                                                   |

[^ds-save]: Since the state of a dataset is continously persisted it cannot be saved.

Additionally, we provide the geometrical primitives
[Vec3Int](webknossos/geometry/vec3_int.html#Vec3Int),
[BoundingBox](webknossos/geometry/bounding_box.html#BoundingBox) and
[Mag](webknossos/geometry/mag.html#Mag).

webKnossos user information can be retrieved via the [`User`](webknossos/client/user.html#User) class.
"""

from webknossos.annotation import *
from webknossos.client import *
from webknossos.dataset import *
from webknossos.geometry import *
from webknossos.skeleton import *
