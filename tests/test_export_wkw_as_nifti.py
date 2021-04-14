from wkcuber.api.Dataset import WKDataset
from wkcuber.api.bounding_box import BoundingBox
import os

from wkcuber.export_wkw_as_nifti import run
from wkcuber.mag import Mag
import numpy as np
import nibabel as nib

ds_name = "simple_wk_dataset"
source_path = os.path.join("testdata", ds_name)


def test_export_nifti_file() -> None:
    destination_path = os.path.join("testoutput", ds_name + "_nifti")
    bbox = BoundingBox((100, 100, 10), (100, 500, 50))
    bbox_dict = bbox.as_config()
    args_list = [
        "--source_path",
        source_path,
        "--destination_path",
        destination_path,
        "--name",
        "test_export",
        "--bbox",
        bbox.as_csv(),
        "--mag",
        "1",
    ]

    run(args_list)

    wk_ds = WKDataset(source_path)

    layer_names = wk_ds.layers

    for layer_name in layer_names:
        wk_mag = wk_ds.get_layer(layer_name).get_mag(Mag(1))
        correct_image = wk_mag.read(bbox_dict["topleft"], bbox_dict["size"])
        # nifti is transposed
        correct_image = correct_image.transpose(1, 2, 3, 0)
        correct_image = np.squeeze(correct_image)

        nifti_path = os.path.join(destination_path, f"test_export{layer_name}.nii")

        assert os.path.isfile(
            nifti_path
        ), f"Expected a nifti to be written at: {nifti_path}."

        nifti = nib.load(nifti_path)
        test_image = np.array(nifti.get_fdata())

        assert np.array_equal(correct_image, test_image), (
            f"The nifti file {nifti_path} that was written is not "
            f"equal to the original wkw_file."
        )


if __name__ == "__main__":
    test_export_nifti_file()
