from time import gmtime, strftime

from skimage import data

import webknossos as wk


def main() -> None:
    with wk.webknossos_context(url="http://localhost:9000", token="secretScmBoyToken"):
        img = data.cell()
        time_str = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
        name = f"cell_{time_str}"
        ds = wk.Dataset.create(name, scale=(107, 107, 107))
        layer = ds.add_layer(
            "color",
            "color",
            dtype_per_layer=img.dtype,
        )
        # add channel and z dimensions and put X before Y,
        # resulting dimensions are C, X, Y, Z.
        layer.add_mag(1, compress=True).write(img.T[None, :, :, None])
        url = ds.upload()
        print(f"Successfully uploaded {url}")


if __name__ == "__main__":
    main()
