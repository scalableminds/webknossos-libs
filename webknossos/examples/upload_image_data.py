from time import gmtime, strftime

import numpy as np
from skimage import data

import webknossos as wk
from webknossos.dataset import COLOR_CATEGORY

# Get your auth token from https://webknossos.org/auth/token
with wk.webknossos_context(url="http://localhost:9000", token="secretToken"):

    # load your data - we use a an example 3D dataset here
    img = data.cells3d()  # (z, c, y, x)

    # make sure that the dimension of your data has the right order
    # we expect the following dimensions Channels, X, Y, Z.
    img = np.transpose(img, [1, 3, 2, 0])

    # choose a name for our dataset
    time_str = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
    name = f"cell_{time_str}"

    # scale is defined in nm
    ds = wk.Dataset.create(name, scale=(290, 260, 260))
    layer = ds.add_layer(
        "raw_microscopy_layer",
        COLOR_CATEGORY,
        dtype_per_layer=img.dtype,
    )

    layer.add_mag(1, compress=True).write(img)

    url = ds.upload()
    print(f"Successfully uploaded {url}")
