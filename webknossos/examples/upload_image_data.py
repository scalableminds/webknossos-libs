from time import gmtime, strftime

from skimage import data

from webknossos.client.context import webknossos_context
from webknossos.dataset.dataset import Dataset

with webknossos_context(url="http://localhost:9000", token="secretScmBoyToken"):
    img = data.cell()
    time_str = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
    name = f"cell_{time_str}"
    ds = Dataset.create(name, scale=(107, 107, 107))
    layer = ds.add_layer(
        "color",
        "color",
        dtype_per_layer=img.dtype,
    )
    layer.add_mag(1, compress=True).write(img.T[None, :, :, None])
    url = ds.upload()
    print(f"Successfully uploaded {url}")
