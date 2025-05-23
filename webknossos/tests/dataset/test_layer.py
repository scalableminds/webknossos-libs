from pathlib import Path
from typing import Tuple, Any

import numpy as np
import tensorstore
import pytest # Import pytest
from skimage.transform import AffineTransform # Import AffineTransform

import webknossos as wk
from webknossos import BoundingBox, Dataset, Layer, Mag # Import necessary classes


# Helper function to create a sample dataset with a color layer
def create_dataset_with_color_layer(
    tmp_path: Path,
    dataset_name: str,
    layer_name: str,
    bounding_box_shape: Tuple[int, int, int],
    dtype: Any = np.uint8,
    num_channels: int = 1,
    voxel_size: Tuple[int, int, int] = (1, 1, 1),
    mag: Mag = Mag(1),
    data_format: str = "raw", # Default to raw for simplicity in tests
    chunk_shape: Tuple[int, int, int] = (64, 64, 64),
) -> Layer:
    """Helper function to create a dataset with a single color layer."""
    dataset_path = tmp_path / dataset_name
    dataset = Dataset(
        dataset_path,
        voxel_size=voxel_size,
        exist_ok=True, # Allow re-creation for tests if needed
    )
    layer = dataset.add_layer(
        layer_name,
        wk.COLOR_CATEGORY,
        bounding_box=BoundingBox((0, 0, 0), shape=bounding_box_shape),
        dtype_per_channel=dtype,
        num_channels=num_channels,
        data_format=data_format,
    )
    # Add the default mag if it doesn't exist, configure with chunk_shape
    if mag not in layer.mags:
        layer.add_mag(mag, chunk_shape=chunk_shape)
    else:
        # Ensure existing mag also has compatible chunk_shape if we were to rely on it
        # For simplicity, we assume add_mag handles or we create fresh datasets.
        pass
    return layer


class TestLayerTransform:
    def test_identity_transform(self, tmp_path: Path) -> None:
        """Tests the transform method with an identity transformation."""
        input_dataset_name = "input_ds_identity"
        input_layer_name = "input_layer_identity"
        output_dataset_name = "output_ds_identity"
        output_layer_name = "output_layer_identity"
        bbox_shape = (128, 128, 128) # Make it a bit larger to test chunking
        dtype = np.uint8
        mag_level = Mag(1)
        chunk_s = (32,32,32) # Smaller chunk for testing multiple chunks

        # 1. Create an input layer with some data
        input_layer = create_dataset_with_color_layer(
            tmp_path,
            input_dataset_name,
            input_layer_name,
            bbox_shape,
            dtype=dtype,
            mag=mag_level,
            chunk_shape=chunk_s,
        )
        # Populate with some data
        # Data shape (C, X, Y, Z)
        input_data = np.random.randint(0, 255, size=(input_layer.num_channels, *bbox_shape), dtype=dtype)
        input_mag_view = input_layer.get_mag(mag_level)
        # Write data to the entire bounding box of the mag view
        input_mag_view.write(input_data, input_mag_view.bounding_box)


        # 2. Create an empty output layer with the same properties
        output_layer = create_dataset_with_color_layer(
            tmp_path,
            output_dataset_name,
            output_layer_name,
            bbox_shape,
            dtype=dtype,
            mag=mag_level,
            chunk_shape=chunk_s,
        )

        # 3. Define an identity inverse_transform function
        def identity_inverse_transform(coords: np.ndarray) -> np.ndarray:
            # coords are (N, 3) in global output space
            # For identity, input global space is the same
            return coords

        # 4. Call layer.transform with the identity transform
        # Use a small number of threads for testing to avoid overhead issues on CI
        input_layer.transform(
            output_layer=output_layer,
            inverse_transform=identity_inverse_transform,
            mag=mag_level,
            num_threads=2, # Use 2 threads for testing parallelism
            chunk_shape=chunk_s # Pass the same chunk_shape
        )

        # 5. Assert that the data in the output layer is identical to the input layer
        output_mag_view = output_layer.get_mag(mag_level)
        output_data = output_mag_view.read(output_mag_view.bounding_box)

        np.testing.assert_array_equal(output_data, input_data,
                                      err_msg="Data in output layer does not match input layer after identity transform.")

        # Additional check: ensure bounding boxes were handled correctly
        assert output_layer.bounding_box == input_layer.bounding_box
        assert output_mag_view.bounding_box.shape == input_mag_view.bounding_box.shape
        assert output_mag_view.bounding_box.min_coord == input_mag_view.bounding_box.min_coord

    def test_translation_transform(self, tmp_path: Path) -> None:
        """Tests the transform method with a translation."""
        input_dataset_name = "input_ds_translate"
        input_layer_name = "input_layer_translate"
        output_dataset_name = "output_ds_translate"
        output_layer_name = "output_layer_translate"
        
        bbox_shape_orig = (64, 64, 64) # Original data size
        dtype = np.uint8
        mag_level = Mag(1)
        chunk_s = (16, 16, 16) # Smaller chunk for testing
        num_ch = 1

        # Translation vector (in global coordinates)
        translation_vector = np.array([10, -5, 20]) # dx, dy, dz

        # 1. Create an input layer with some data
        input_layer = create_dataset_with_color_layer(
            tmp_path, input_dataset_name, input_layer_name,
            bbox_shape_orig, dtype=dtype, mag=mag_level, chunk_shape=chunk_s, num_channels=num_ch
        )
        # Populate with sequential data for easier verification
        input_data_flat = np.arange(np.prod(bbox_shape_orig), dtype=dtype).reshape(bbox_shape_orig)
        input_data = input_data_flat[np.newaxis, ...] # Add channel dimension: (1, X, Y, Z)
        input_mag_view = input_layer.get_mag(mag_level)
        input_mag_view.write(input_data, input_mag_view.bounding_box)

        # 2. Create an empty output layer with the same properties & original bbox
        # The transform method itself doesn't change the output layer's bbox,
        # it writes within the output_layer.bounding_box or the specified output_bounding_box.
        # For a pure translation, the output data region will be the same size as input.
        output_layer = create_dataset_with_color_layer(
            tmp_path, output_dataset_name, output_layer_name,
            bbox_shape_orig, dtype=dtype, mag=mag_level, chunk_shape=chunk_s, num_channels=num_ch
        )

        # 3. Define the inverse_transform for translation
        # output_coord -> input_coord. So, if output is translated by T, input is output - T.
        inverse_translation = lambda coords: coords - translation_vector

        # 4. Call layer.transform
        input_layer.transform(
            output_layer=output_layer,
            inverse_transform=inverse_translation,
            mag=mag_level,
            num_threads=1, # Test with single thread first
            chunk_shape=chunk_s
        )

        # 5. Manually compute the expected output data
        expected_output_data = np.zeros_like(input_data) # Fill with 0 (default fill value)
        
        # Determine the overlapping region in the output layer's perspective
        # These are slices for the output_data array
        out_x_slice = slice(max(0, int(translation_vector[0])), min(bbox_shape_orig[0], int(bbox_shape_orig[0] + translation_vector[0])))
        out_y_slice = slice(max(0, int(translation_vector[1])), min(bbox_shape_orig[1], int(bbox_shape_orig[1] + translation_vector[1])))
        out_z_slice = slice(max(0, int(translation_vector[2])), min(bbox_shape_orig[2], int(bbox_shape_orig[2] + translation_vector[2])))

        # Determine the corresponding region in the input layer's perspective
        # These are slices for the input_data array
        in_x_slice = slice(max(0, int(-translation_vector[0])), min(bbox_shape_orig[0], int(bbox_shape_orig[0] - translation_vector[0])))
        in_y_slice = slice(max(0, int(-translation_vector[1])), min(bbox_shape_orig[1], int(bbox_shape_orig[1] - translation_vector[1])))
        in_z_slice = slice(max(0, int(-translation_vector[2])), min(bbox_shape_orig[2], int(bbox_shape_orig[2] - translation_vector[2])))
        
        expected_output_data[0, out_x_slice, out_y_slice, out_z_slice] = input_data[0, in_x_slice, in_y_slice, in_z_slice]

        output_mag_view = output_layer.get_mag(mag_level)
        actual_output_data = output_mag_view.read(output_mag_view.bounding_box)
        
        np.testing.assert_array_equal(actual_output_data, expected_output_data,
                                      err_msg="Data in output layer does not match expected translated data.")

    def test_scaling_transform(self, tmp_path: Path) -> None:
        """Tests the transform method with a scaling transformation (2x magnification)."""
        input_dataset_name = "input_ds_scale"
        input_layer_name = "input_layer_scale"
        output_dataset_name = "output_ds_scale"
        output_layer_name = "output_layer_scale"
        
        input_bbox_shape = (32, 32, 32) # Small input for easier manual verification
        dtype = np.uint16 # Use a different dtype
        mag_level = Mag(1) # Assume scaling happens at mag 1 for simplicity
        chunk_s = (16, 16, 16)
        num_ch = 1
        scaling_factor = 2.0

        # 1. Create input layer and populate with data
        input_layer = create_dataset_with_color_layer(
            tmp_path, input_dataset_name, input_layer_name,
            input_bbox_shape, dtype=dtype, mag=mag_level, chunk_shape=chunk_s, num_channels=num_ch
        )
        # Create simple ramp data for predictable scaling
        input_data_flat = np.arange(np.prod(input_bbox_shape), dtype=dtype).reshape(input_bbox_shape)
        input_data = input_data_flat[np.newaxis, ...] # (1, X, Y, Z)
        input_mag_view = input_layer.get_mag(mag_level)
        input_mag_view.write(input_data, input_mag_view.bounding_box)

        # 2. Define output bounding box and create output layer
        # Output bounding box should be scaled version of input
        output_bbox_shape = tuple(int(s * scaling_factor) for s in input_bbox_shape)
        output_layer = create_dataset_with_color_layer(
            tmp_path, output_dataset_name, output_layer_name,
            output_bbox_shape, dtype=dtype, mag=mag_level, chunk_shape=chunk_s, num_channels=num_ch
        )
        # The output_bounding_box for the transform call will be the output_layer's bounding_box.

        # 3. Define inverse_transform for scaling
        # output_coord -> input_coord. So, input_coord = output_coord / scaling_factor
        inverse_scaling = lambda coords: coords / scaling_factor

        # 4. Call layer.transform
        input_layer.transform(
            output_layer=output_layer,
            inverse_transform=inverse_scaling,
            mag=mag_level,
            num_threads=0, # Test with num_threads=0 (sequential)
            chunk_shape=chunk_s
            # output_bounding_box is implicitly output_layer.bounding_box here
        )

        # 5. Manually compute expected output data (nearest neighbor)
        expected_output_data = np.zeros((num_ch, *output_bbox_shape), dtype=dtype)
        for c in range(num_ch):
            for ox in range(output_bbox_shape[0]):
                for oy in range(output_bbox_shape[1]):
                    for oz in range(output_bbox_shape[2]):
                        # Corresponding input coordinate (float)
                        ix_f = ox / scaling_factor
                        iy_f = oy / scaling_factor
                        iz_f = oz / scaling_factor
                        
                        # Nearest neighbor rounding
                        ix = int(round(ix_f))
                        iy = int(round(iy_f))
                        iz = int(round(iz_f))
                        
                        # Clamp to input bounds
                        ix = np.clip(ix, 0, input_bbox_shape[0] - 1)
                        iy = np.clip(iy, 0, input_bbox_shape[1] - 1)
                        iz = np.clip(iz, 0, input_bbox_shape[2] - 1)
                        
                        expected_output_data[c, ox, oy, oz] = input_data[c, ix, iy, iz]
        
        output_mag_view = output_layer.get_mag(mag_level)
        actual_output_data = output_mag_view.read(output_mag_view.bounding_box)

        np.testing.assert_array_equal(actual_output_data, expected_output_data,
                                      err_msg="Data in output layer does not match expected scaled data.")

    def test_affine_transform_simple_rotation(self, tmp_path: Path) -> None:
        """Tests a simple 90-degree rotation around Z-axis, followed by translation."""
        input_name, output_name = "input_ds_affine", "output_ds_affine"
        layer_name = "layer_affine"
        # Input is a 2x1x1 strip along X to make rotation easy to verify
        # Centered at (0.5, 0.5, 0.5) before translation for rotation
        input_bbox_shape = (2, 1, 1) 
        dtype, mag_level, num_ch = np.uint8, Mag(1), 1
        chunk_s = (1, 1, 1)

        input_layer = create_dataset_with_color_layer(
            tmp_path, input_name, layer_name, input_bbox_shape,
            dtype=dtype, mag=mag_level, chunk_shape=chunk_s, num_channels=num_ch
        )
        # Data: [value1, value2] along X axis for the single channel
        input_data = np.array([[[[100]], [[200]]]]).astype(dtype).reshape(num_ch, *input_bbox_shape) # (C,X,Y,Z)
        input_mag_view = input_layer.get_mag(mag_level)
        input_mag_view.write(input_data, input_mag_view.bounding_box)

        # Output layer will be large enough to contain the rotated & translated data
        # Original data at (0,0,0) and (1,0,0)
        # Rotated 90 deg around Z (about origin 0,0): (0,0,0) -> (0,0,0); (1,0,0) -> (0,1,0)
        # Translated by (1,1,0): (0,0,0) -> (1,1,0); (0,1,0) -> (1,2,0)
        # So, output needs to cover at least (1,1,0) to (1,2,0)
        output_bbox_shape = (2, 3, 1) # Make it a bit larger to be safe
        output_layer = create_dataset_with_color_layer(
            tmp_path, output_name, layer_name, output_bbox_shape,
            dtype=dtype, mag=mag_level, chunk_shape=chunk_s, num_channels=num_ch
        )
        
        # Define affine transform: 90 deg rotation around Z ( skimage uses degrees) then translate by (1,1,0)
        # skimage AffineTransform works with (col, row, z) which is (x, y, z)
        # Rotation matrix for 90 deg around Z: [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        # Translation: (1,1,0)
        # Center of rotation is implicitly (0,0,0) for the matrix part.
        # The inverse_transform in the method expects global coords.
        # Our input data is at global coords (0,0,0) and (1,0,0).
        
        # Transformation: T_translate * T_rotate
        # Output_coord = T_translate * T_rotate * Input_coord
        # Input_coord = T_rotate_inv * T_translate_inv * Output_coord
        
        # Rotation by +90 deg around Z (counter-clockwise)
        # x' = x*cos(a) - y*sin(a)
        # y' = x*sin(a) + y*cos(a)
        # For +90: x' = -y, y' = x
        # Inverse rotation (-90 deg): x = y', y = -x'
        
        # Translation by (tx, ty, tz)
        # x_out = x_rot + tx
        # y_out = y_rot + ty
        # Inverse translation: x_rot = x_out - tx, y_rot = y_out - ty

        # Combined inverse:
        # 1. Output coord (ox, oy, oz)
        # 2. Inverse translate: (ox - tx, oy - ty, oz - tz) -> (ix_t, iy_t, iz_t)
        # 3. Inverse rotate: (iy_t, -ix_t, iz_t) -> input coord (inx, iny, inz)
        
        tx, ty, tz = 1, 1, 0
        
        def affine_inverse_transform(output_coords_global: np.ndarray) -> np.ndarray:
            # output_coords_global is (N,3)
            input_coords_translated_inv = output_coords_global - np.array([tx, ty, tz])
            
            input_coords_rotated_inv = np.zeros_like(input_coords_translated_inv)
            input_coords_rotated_inv[:, 0] = input_coords_translated_inv[:, 1]  # x_in = y_translated_inv
            input_coords_rotated_inv[:, 1] = -input_coords_translated_inv[:, 0] # y_in = -x_translated_inv
            input_coords_rotated_inv[:, 2] = input_coords_translated_inv[:, 2]  # z_in = z_translated_inv
            return input_coords_rotated_inv

        input_layer.transform(
            output_layer, affine_inverse_transform, mag=mag_level, num_threads=None, chunk_shape=chunk_s
        )

        expected_output = np.zeros((num_ch, *output_bbox_shape), dtype=dtype)
        # Input (0,0,0) [val 100] -> Rot (0,0,0) -> Trans (1,1,0)
        # Input (1,0,0) [val 200] -> Rot (0,1,0) -> Trans (1,2,0)
        if 0 <= 1 < output_bbox_shape[0] and 0 <= 1 < output_bbox_shape[1] and 0 <= 0 < output_bbox_shape[2]:
             expected_output[0, 1, 1, 0] = 100 # voxel at (1,1,0) in output
        if 0 <= 1 < output_bbox_shape[0] and 0 <= 2 < output_bbox_shape[1] and 0 <= 0 < output_bbox_shape[2]:
             expected_output[0, 1, 2, 0] = 200 # voxel at (1,2,0) in output

        actual_output = output_layer.get_mag(mag_level).read()
        np.testing.assert_array_equal(actual_output, expected_output, err_msg="Affine transform failed.")

    def test_output_bounding_box_smaller_and_shifted(self, tmp_path: Path) -> None:
        """Tests transform when output_bounding_box is smaller and shifted."""
        input_ds, input_layer_n = "in_ds_obb", "in_l_obb"
        output_ds, output_layer_n = "out_ds_obb", "out_l_obb"
        input_bbox_shape = (50, 50, 50)
        dtype, mag, num_ch = np.uint8, Mag(1), 1
        chunk_s = (10,10,10)

        input_layer = create_dataset_with_color_layer(
            tmp_path, input_ds, input_layer_n, input_bbox_shape,
            dtype=dtype, mag=mag, chunk_shape=chunk_s, num_channels=num_ch
        )
        # Sequential data
        input_data = np.arange(np.prod(input_bbox_shape), dtype=dtype).reshape(num_ch, *input_bbox_shape)
        input_layer.get_mag(mag).write(input_data, input_layer.get_mag(mag).bounding_box)

        # Output layer is larger, but we'll only write to a small, shifted part of it
        output_layer_bbox_shape = (60, 60, 60)
        output_layer = create_dataset_with_color_layer(
            tmp_path, output_ds, output_layer_n, output_layer_bbox_shape,
            dtype=dtype, mag=mag, chunk_shape=chunk_s, num_channels=num_ch
        )
        # Fill output layer with a distinct value to check only target OBB is written
        fill_value = 77
        output_layer.get_mag(mag).write(np.full((num_ch, *output_layer_bbox_shape), fill_value, dtype=dtype))


        # Define the specific output_bounding_box for the transform operation
        # This OBB is in global coordinates.
        # Let's pick a 20x20x20 cube shifted by (5,5,5) in the output layer
        # This OBB will read from input layer starting at (5,5,5) due to identity transform
        obb_min_coord = (5, 5, 5)
        obb_shape = (20, 20, 20)
        specific_output_bb = BoundingBox(min_coord=obb_min_coord, shape=obb_shape)

        input_layer.transform(
            output_layer,
            inverse_transform=lambda coords: coords, # Identity
            mag=mag,
            output_bounding_box=specific_output_bb,
            num_threads=2,
            chunk_shape=chunk_s
        )

        # Expected data in the output layer
        expected_data_in_output_layer = np.full((num_ch, *output_layer_bbox_shape), fill_value, dtype=dtype)
        # The part that should be overwritten comes from input_data[0, 5:25, 5:25, 5:25]
        # and written to expected_data_in_output_layer[0, 5:25, 5:25, 5:25]
        src_data_slice = (slice(None), slice(obb_min_coord[0], obb_min_coord[0]+obb_shape[0]), \
                                       slice(obb_min_coord[1], obb_min_coord[1]+obb_shape[1]), \
                                       slice(obb_min_coord[2], obb_min_coord[2]+obb_shape[2]))
        expected_data_in_output_layer[src_data_slice] = input_data[src_data_slice]
        
        actual_output_data = output_layer.get_mag(mag).read()
        np.testing.assert_array_equal(actual_output_data, expected_data_in_output_layer)

    def test_transform_all_coords_outside_input(self, tmp_path: Path) -> None:
        """Tests transform when all transformed coords are outside input layer's bounds."""
        input_ds, input_layer_n = "in_ds_outside", "in_l_outside"
        output_ds, output_layer_n = "out_ds_outside", "out_l_outside"
        bbox_shape = (10, 10, 10)
        dtype, mag, num_ch, chunk_s = np.uint8, Mag(1), 1, (5,5,5)

        input_layer = create_dataset_with_color_layer(
            tmp_path, input_ds, input_layer_n, bbox_shape, dtype=dtype, mag=mag, chunk_shape=chunk_s
        )
        input_layer.get_mag(mag).write(np.ones((num_ch, *bbox_shape), dtype=dtype)) # Fill with 1s

        output_layer = create_dataset_with_color_layer(
            tmp_path, output_ds, output_layer_n, bbox_shape, dtype=dtype, mag=mag, chunk_shape=chunk_s
        )
        
        # Translation that shifts everything out of input bounds
        # Input is at [0,0,0] to [9,9,9]. Shift by [100,100,100]
        translation_far = np.array([100, 100, 100])
        inverse_transform_far = lambda coords: coords - translation_far

        input_layer.transform(
            output_layer, inverse_transform_far, mag=mag, num_threads=1, chunk_shape=chunk_s
        )
        
        # Expected output is all zeros (default fill value from clamping/empty read)
        expected_output = np.zeros((num_ch, *bbox_shape), dtype=dtype)
        actual_output = output_layer.get_mag(mag).read()
        np.testing.assert_array_equal(actual_output, expected_output)

    def test_transform_small_input_layer(self, tmp_path: Path) -> None:
        """Tests transform with a very small input layer (2x2x2)."""
        input_ds, input_layer_n = "in_ds_small", "in_l_small"
        output_ds, output_layer_n = "out_ds_small", "out_l_small"
        input_bbox_shape = (2, 2, 2)
        dtype, mag, num_ch, chunk_s = np.uint8, Mag(1), 1, (1,1,1) # Chunk size 1

        input_layer = create_dataset_with_color_layer(
            tmp_path, input_ds, input_layer_n, input_bbox_shape, dtype=dtype, mag=mag, chunk_shape=chunk_s
        )
        input_data = np.arange(np.prod(input_bbox_shape), dtype=dtype).reshape(num_ch, *input_bbox_shape)
        input_layer.get_mag(mag).write(input_data, input_layer.get_mag(mag).bounding_box)

        output_layer = create_dataset_with_color_layer(
            tmp_path, output_ds, output_layer_n, input_bbox_shape, dtype=dtype, mag=mag, chunk_shape=chunk_s
        )
        
        # Simple identity transform
        input_layer.transform(
            output_layer, lambda coords: coords, mag=mag, num_threads=None, chunk_shape=chunk_s
        )
        
        actual_output = output_layer.get_mag(mag).read()
        np.testing.assert_array_equal(actual_output, input_data)


def test_add_mag_from_zarrarray(tmp_path: Path) -> None:
    dataset = wk.Dataset(
        tmp_path / "test_add_mag_from_zarrarray", voxel_size=(10, 10, 10)
    )
    layer = dataset.add_layer(
        "color",
        wk.COLOR_CATEGORY,
        data_format="zarr3",
        bounding_box=wk.BoundingBox((0, 0, 0), (16, 16, 16)),
    )
    zarr_mag_path = tmp_path / "zarr_data" / "mag1.zarr"
    zarr_data = np.random.randint(0, 255, (16, 16, 16), dtype="uint8")
    zarr_mag = tensorstore.open(
        {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": str(zarr_mag_path)},
            "metadata": {
                "data_type": "uint8",
                "shape": (16, 16, 16),
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": (8, 8, 8)},
                },
                "chunk_key_encoding": {
                    "name": "default",
                    "configuration": {"separator": "."},
                },
                "fill_value": 0,
                "codecs": [
                    {
                        "name": "bytes",
                        "configuration": {"endian": "little"},
                    },
                    {
                        "name": "blosc",
                        "configuration": {
                            "cname": "zstd",
                            "clevel": 5,
                            "shuffle": "shuffle",
                            "typesize": 1,
                        },
                    },
                ],
            },
            "create": True,
        }
    ).result()

    zarr_mag[:].write(zarr_data).result()

    layer.add_mag_from_zarrarray("1", zarr_mag_path, extend_layer_bounding_box=False)

    assert layer.get_mag("1").read().shape == (1, 16, 16, 16)
    assert layer.get_mag("1").info.num_channels == 1
    assert layer.get_mag("1").info.dimension_names == ("c", "x", "y", "z")
    assert (layer.get_mag("1").read()[0] == zarr_data).all()
