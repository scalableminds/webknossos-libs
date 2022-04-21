from time import gmtime, strftime

import numpy as np
from skimage import data

import webknossos as wk
from webknossos.dataset import COLOR_CATEGORY
from webknossos.dataset.properties import (
    DatasetViewConfiguration,
    LayerViewConfiguration,
)


def main() -> None:
    # load your data - we use an example 3D dataset here
    img = data.cells3d()  # (z, c, y, x)

    # make sure that the dimension of your data has the right order
    # we expect the following dimensions: Channels, X, Y, Z.
    img = np.transpose(img, [1, 3, 2, 0])

    # choose a name for our dataset
    time_str = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
    name = f"cell_{time_str}"

    # scale is defined in nm
    ds = wk.Dataset(name, scale=(260, 260, 290))

    ds.default_view_configuration = DatasetViewConfiguration(zoom=0.35)

    # The example microscopy data has two channels
    # Channel 0 contains cell membranes, channel 1 contains nuclei.
    layer_membranes = ds.add_layer(
        "cell membranes",
        COLOR_CATEGORY,
        dtype_per_layer=img.dtype,
    )

    layer_membranes.add_mag(1, compress=True).write(img[0, :])

    layer_membranes.default_view_configuration = LayerViewConfiguration(
        color=(17, 212, 17), intensity_range=(0, 16000)
    )

    layer_nuclei = ds.add_layer(
        "nuclei",
        COLOR_CATEGORY,
        dtype_per_layer=img.dtype,
    )

    layer_nuclei.add_mag(1, compress=True).write(img[1, :])

    layer_nuclei.default_view_configuration = LayerViewConfiguration(
        color=(212, 17, 17), intensity_range=(3000, 30000)
    )

    url = ds.upload()
    print(f"Successfully uploaded {url}")


if __name__ == "__main__":
    main()
