from pathlib import Path

import nibabel as nib
import numpy as np
from webknossos import BoundingBox, Dataset, Mag
from wkcuber.export_wkw_as_nifti import export_wkw_as_nifti_from_arg_list

from .constants import TESTDATA_DIR

DS_NAME = "simple_wkw_dataset"
SOURCE_PATH = TESTDATA_DIR / DS_NAME


def test_export_nifti_file(tmp_path: Path) -> None:
    destination_path = tmp_path / f"{DS_NAME}_nifti"
    destination_path.mkdir()

    bbox = BoundingBox((100, 100, 10), (100, 500, 50))
    bbox_dict = bbox.to_config_dict()
    args_list = [
        "--source_path",
        str(SOURCE_PATH),
        "--destination_path",
        str(destination_path),
        "--name",
        "test_export",
        "--source_bbox",
        bbox.to_csv(),
        "--mag",
        "1",
    ]

    export_wkw_as_nifti_from_arg_list(args_list)

    wk_ds = Dataset.open(SOURCE_PATH)

    for layer_name, layer in wk_ds.layers.items():
        correct_image = layer.get_mag(Mag(1)).read(
            bbox_dict["topleft"], bbox_dict["size"]
        )
        # nifti is transposed
        correct_image = correct_image.transpose(1, 2, 3, 0)
        correct_image = np.squeeze(correct_image)

        nifti_path = destination_path.joinpath(f"test_export_{layer_name}.nii")

        assert nifti_path.is_file(), f"Expected a nifti to be written at: {nifti_path}."

        nifti = nib.load(str(nifti_path))
        test_image = np.array(nifti.get_fdata())

        assert np.array_equal(correct_image, test_image), (
            f"The nifti file {nifti_path} that was written is not "
            f"equal to the original wkw_file."
        )
