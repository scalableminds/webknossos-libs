"""
# Dataset API

The high-level dataset API automatically reads and writes meta information for any dataset and updates them if necessary, such as the `datasource-properties.json`.

A dataset (`wkcuber.api.dataset.Dataset`) is the entry-point for this API.
The dataset stores the data on disk in `.wkw`-files (see [webknossos-wrap (wkw)](https://github.com/scalableminds/webknossos-wrap)).

Each dataset consists of one or more layers (wkcuber.api.layer.Layer), which themselves can comprise multiple magnifications (wkcuber.api.mag_view.MagView).

## Examples
### Opening Datasets
```python
from wkcuber.api import Dataset

dataset = Dataset(<path_to_dataset>)
# Assuming that the dataset has a layer called 'color' and the layer has the magnification "1"
layer = dataset.get_layer("color")
mag1 = layer.get_mag("1")
```

### Creating Datasets
```python
from wkcuber.api import Dataset
from wkcuber.api import Layer

dataset = Dataset.create(<path_to_new_dataset>, scale=(1, 1, 1))
layer = dataset.add_layer(
    layer_name="color",
    category=LayerCategories.COLOR_TYPE,
    dtype_per_channel="uint8",
    num_channels=3
)

mag1 = layer.add_mag("1")
```

### Reading Datasets
```python
from wkcuber.api import Dataset

dataset = Dataset(<path_to_dataset>)
# Assuming that the dataset has a layer called 'color' and the layer has the magnification "1" and "2"
layer = dataset.get_layer("color")
mag1 = layer.get_mag("1")
mag2 = layer.get_mag("2")

data_in_mag1 = mag1.read()  # the offset and size from the properties are used
data_in_mag1_subset = mag1.read(offset=(10, 20, 30), size=(512, 512, 32))

data_in_mag2 = mag2.read()  # the offset and size from the properties are used
data_in_mag2_subset = mag2.read(offset=(5, 10, 15), size=(256, 256, 16))
```

### Writing Datasets
```python
from wkcuber.api import Dataset

dataset = Dataset(<path_to_dataset>)
# Assuming that the dataset has a layer called 'color' and the layer has the magnification "1" and "2"
layer = dataset.get_layer("color")
mag1 = layer.get_mag("1")
mag2 = layer.get_mag("2")

# The properties are updated, if the written data exceeds the bounding box in the properties
mag1.write(
    offset=(10, 20, 30),
    data=(np.random.rand(3, 512, 512, 32) * 255).astype(np.uint8)  # assuming the layer has 3 channels
)

mag2.write(
    offset=(5, 10, 15),
    data=(np.random.rand(3, 256, 256, 16) * 255).astype(np.uint8)  # assuming the layer has 3 channels
)
```
"""

from .bounding_box import BoundingBox
from .dataset import Dataset
from .layer import Layer
from .mag_view import MagView
from .view import View
