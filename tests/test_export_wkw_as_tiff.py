from wkcuber.export_wkw_as_tiff import export_tiff_stack, wkw_name_and_bbox_to_tiff_name, create_parser
import os
from PIL import Image
from wkcuber.mag import Mag
import wkw
import numpy as np

def test_export_tiff_stack():
    parser = create_parser()
    args_list = ["--source_path", os.path.join("testdata", "WT1_wkw"),
                 "--destination_path", os.path.join("testoutput", "WT1_wkw"),
                 "--layer_name", "color",
                 "--name", "test_export",
                 "--bbox", "0,0,0,100,100,5",
                 "--mag", "1"]
    args = parser.parse_args(args_list)

    bbox = [int(s.strip()) for s in args.bbox.split(",")]
    bbox = {"topleft": bbox[0:3], "size": bbox[3:6]}

    export_tiff_stack(
        wkw_file_path=args.source_path,
        wkw_layer=args.layer_name,
        bbox=bbox,
        mag=Mag(args.mag),
        destination_path=args.destination_path,
        name=args.name,
        tiling_slice_size=None,
        args=args,
    )

    test_wkw_file_path = os.path.join("testdata", "WT1_wkw", "color", Mag(1).to_layer_name())
    with wkw.Dataset.open(test_wkw_file_path) as dataset:
        slice_bbox = bbox
        slice_bbox["size"] = [slice_bbox["size"][0], slice_bbox["size"][1], 1]
        for data_slice_index in range(1, bbox["size"][2] + 1):
            slice_bbox["offset"] = [slice_bbox["topleft"][0], slice_bbox["topleft"][1], bbox["topleft"][2] + data_slice_index]
            tiff_path = os.path.join(args.destination_path, wkw_name_and_bbox_to_tiff_name(args.name, data_slice_index))

            assert os.path.isfile(tiff_path), f"Expected a tiff to be written at: {tiff_path}."

            test_image = np.array(Image.open(tiff_path))
            test_image.transpose((1, 0))

            correct_image = dataset.read(off=slice_bbox["topleft"], shape=slice_bbox["size"])
            correct_image = np.squeeze(correct_image)

            assert np.array_equal(correct_image, test_image), f"The tiff file {tiff_path} that was written is not " \
                                                              f"equal to the original wkw_file."
