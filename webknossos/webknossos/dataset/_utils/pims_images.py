import warnings
from contextlib import contextmanager, nullcontext
from os import PathLike
from typing import Iterator, List, Optional, Sequence, Tuple, TypeVar, Union, cast
from urllib.error import HTTPError

import numpy as np
import pims

from webknossos.dataset.mag_view import MagView
from webknossos.geometry.vec3_int import Vec3Int


class PimsImages:
    def __init__(
        self,
        images: Union[str, "pims.FramesSequence", List[Union[str, PathLike]]],
        timepoint: Optional[int],
        swap_xy: bool,
        flip_x: bool,
        flip_y: bool,
        flip_z: bool,
        use_bioformats: bool,
    ) -> None:
        """
        During initialization the pims objects are examined and configured to produce
        ndarrays that follow the following form:
        (self._iter_dim, *self._img_dims)
        self._iter_dim can be either "z", "t" or "" if the image is 2D.
        In the latter case, the inner 2D image is still wrapped in a single-element list
        by _open_images() to be consistent with 3D images.
        self._img_dims can consist of "x", "y" and "c", where "c" is optional and must be
        at the start or the end, so one of "xy", "yx", "xyc", "yxc", "cxy", "cyx".

        The part "IDENTIFY AXIS ORDER" figures out (self._iter_dim, *self._img_dims)
        from out-of-the-box pims images. Afterwards self._open_images() produces
        images consistent with those variables.

        The part "IDENTIFY SHAPE & CHANNELS" uses this information and the well-defined
        images to figure out the shape & num_channels.
        """
        ## we use images as the name for the entered contextmanager,
        ## the `del` prevents any confusion with the passed argument.
        self._original_images = images
        del images

        ## arguments as inner attributes
        self._timepoint = timepoint
        self._swap_xy = swap_xy
        self._flip_x = flip_x
        self._flip_y = flip_y
        self._flip_z = flip_z
        self._use_bioformats = use_bioformats

        ## attributes that will be set in __init__()
        self._iter_dim = None
        # _img_dims

        ## attributes only for pims.FramesSequenceND instances:
        # _default_coords
        # _init_c_axis

        ## attributes that will also be set in __init__()
        # dtype
        # expected_shape
        # num_channels
        # _first_n_channels

        #######################
        # IDENTIFY AXIS ORDER #
        #######################

        with self._open_images() as images:
            assert isinstance(
                images, pims.FramesSequence
            ), f"{type(images)} does not inherit from pims.FramesSequence"
            self.dtype = images.dtype

            if isinstance(images, pims.FramesSequenceND):
                assert all(
                    axis in "xyzct" for axis in images.axes
                ), f"Found unknown axes {set(images.axes) - set('xyzct')}"

                self._default_coords = {}
                self._init_c_axis = False
                if isinstance(images, pims.imageio_reader.ImageIOReader):
                    # bugfix for ImageIOReader which misses channel axis sometimes,
                    # assuming channels come last:
                    if (
                        len(images.shape) > len(images.sizes)
                        and "c" not in images.sizes
                    ):
                        images._init_axis("c", images.shape[-1])
                        self._init_c_axis = True
                if images.sizes.get("c", 1) > 1:
                    self._img_dims = "cyx"
                else:
                    if "c" in images.axes:
                        self._default_coords["c"] = 0
                    self._img_dims = "yx"

                self._iter_dim = ""

                if images.sizes.get("z", 1) > 1:
                    self._iter_dim = "z"
                elif "z" in images.axes:
                    self._default_coords["z"] = 0

                if timepoint is None:
                    if images.sizes.get("t", 1) > 1:
                        assert self._iter_dim != "z", (
                            f"Found both axes t {images.sizes.get('t', 1) } "
                            + f"and z {images.sizes.get('z', 1) > 1}, "
                            + "cannot use both without setting timepoint"
                        )
                        self._iter_dim = "t"
                    elif "t" in images.axes:
                        self._default_coords["t"] = 0
                else:
                    assert "t" in images.axes
                    self._default_coords["t"] = timepoint
            else:
                if len(images.shape) == 2:
                    self._img_dims = "yx"
                    self._iter_dim = ""
                elif len(images.shape) == 3:
                    if images.shape[2] == 1 or (
                        images.shape[2] == 3 and images.dtype == np.dtype("uint8")
                    ):
                        self._img_dims = "yxc"
                        self._iter_dim = ""
                    elif images.shape[0] == 1 or (
                        images.shape[0] == 3 and images.dtype == np.dtype("uint8")
                    ):
                        self._img_dims = "cyx"
                        self._iter_dim = ""
                    else:
                        self._img_dims = "yx"
                        self._iter_dim = "z"
                elif len(images.shape) == 4:
                    if images.shape[1] == 1 or (
                        images.shape[1] == 3 and images.dtype == np.dtype("uint8")
                    ):
                        self._img_dims = "cyx"
                    else:
                        self._img_dims = "yxc"
                    self._iter_dim = "z"
                else:
                    raise RuntimeError(
                        f"Got {len(images.shape)} axes for the images, "
                        + "cannot map to 3D+channels, consider setting timepoint."
                    )

        #############################
        # IDENTIFY SHAPE & CHANNELS #
        #############################

        with self._open_images() as images:
            if isinstance(images, list):
                images_shape = (len(images),) + cast(
                    pims.FramesSequence, images[0]
                ).shape
            else:
                images_shape = images.shape
            c_index = self._img_dims.find("c")
            if c_index == -1:
                self.num_channels = 1
            else:
                # Since images_shape contains the first dimension iter_dim,
                # we need to offset the index by one before accessing the images_shape.
                # images_shape corresponds to (z, *_img_dims)
                self.num_channels = images_shape[c_index + 1]

            x_index = self._img_dims.find("x") + 1
            y_index = self._img_dims.find("y") + 1
            if swap_xy:
                x_index, y_index = y_index, x_index
            self.expected_shape = Vec3Int(
                images_shape[x_index], images_shape[y_index], images_shape[0]
            )

        self._first_n_channels = None
        if self.num_channels > 3:
            warnings.warn(
                f"Found more than 3 channels ({self.num_channels}), clamping to the first 3."
            )
            self.num_channels = 3
            self._first_n_channels = 3

    @contextmanager
    def _open_images(
        self,
    ) -> Iterator[Union[pims.FramesSequence, List[pims.FramesSequence]]]:
        """
        This yields well-defined images of the form (self._iter_dim, *self._img_dims),
        after IDENTIFY AXIS ORDER of __init__() has run.
        For a 2D image this is achieved by wrapping it in a list.
        """
        with warnings.catch_warnings():

            if isinstance(self._original_images, pims.FramesSequence):
                images_context_manager = nullcontext(enter_result=self._original_images)
            else:
                if self._use_bioformats:
                    # There is a wrong warning about jpype, supressing it here.
                    # See issue https://github.com/soft-matter/pims/issues/384
                    warnings.filterwarnings(
                        "ignore",
                        "Due to an issue with JPype 0.6.0, reading is slower.*",
                        category=UserWarning,
                        module="pims.bioformats",
                    )
                    try:
                        pims.bioformats._find_jar()
                    except HTTPError:
                        # We cannot use the newest bioformats version,
                        # since it does not include the necessary loci_tools.jar.
                        # Updates to support newer bioformats jars with pims are in PR
                        # https://github.com/soft-matter/pims/pull/403
                        pims.bioformats.download_jar(version="6.7.0")
                    if "*" in str(self._original_images) or isinstance(
                        self._original_images, list
                    ):
                        images_context_manager = pims.ReaderSequence(
                            self._original_images, pims.bioformats.BioformatsReader
                        )
                    else:
                        images_context_manager = pims.bioformats.BioformatsReader(
                            self._original_images
                        )
                else:
                    images_context_manager = pims.open(self._original_images)

            with images_context_manager as images:
                if isinstance(images, pims.FramesSequenceND):
                    if hasattr(self, "_img_dims"):
                        # first part of __init__() has happened
                        images.default_coords.update(self._default_coords)
                        if self._init_c_axis and "c" not in images.sizes:
                            # Bugfix for ImageIOReader which misses channel axis sometimes,
                            # assuming channels come last. _init_c_axis is set in __init__().
                            images._init_axis("c", images.shape[-1])
                            for key in list(images._get_frame_dict.keys()):
                                images._get_frame_dict[
                                    key + ("c",)
                                ] = images._get_frame_dict.pop(key)
                        images.bundle_axes = self._img_dims
                        images.iter_axes = self._iter_dim or ""
                else:
                    if self._timepoint is not None:
                        images = images[self._timepoint]
                    if self._iter_dim == "":
                        # add outer list to wrap 2D images as 3D-like structure
                        images = [images]
                yield images

    def copy_to_view(
        self, args: Tuple[int, int], mag_view: MagView, is_segmentation: bool
    ) -> Tuple[Tuple[int, int], Optional[int]]:
        """Copies the images according to the passed arguments to the given mag_view.
        args is expected to be the start and end of the z-range, meant for usage with an executor."""
        z_start, z_end = args
        shapes = []
        max_id: Optional[int]
        if is_segmentation:
            max_id = 0
        else:
            max_id = None

        with self._open_images() as images:
            if self._flip_z:
                images = images[::-1]  # pylint: disable=unsubscriptable-object
            with mag_view.get_buffered_slice_writer(
                absolute_offset=(0, 0, z_start * mag_view.mag.z),
                buffer_size=mag_view.info.chunk_size.z,
            ) as writer:
                for image_slice in images[z_start:z_end]:
                    image_slice = np.array(image_slice)
                    # place channels first
                    if self._img_dims.endswith("c"):
                        image_slice = np.moveaxis(image_slice, source=-1, destination=0)
                    # ensure the last two axes are xy:
                    if ("yx" in self._img_dims and not self._swap_xy) or (
                        "xy" in self._img_dims and self._swap_xy
                    ):
                        image_slice = image_slice.swapaxes(-1, -2)

                    if "c" in self._img_dims:
                        if self._first_n_channels is not None:
                            image_slice = image_slice[: self._first_n_channels]
                        assert image_slice.shape[0] == self.num_channels, (
                            f"Image shape {image_slice.shape} does not fit to the number of channels "
                            + f"{self.num_channels} which are expected in the first axis."
                        )

                    if self._flip_x:
                        image_slice = np.flip(image_slice, -2)
                    if self._flip_y:
                        image_slice = np.flip(image_slice, -1)

                    if max_id is not None:
                        max_id = max(max_id, image_slice.max())
                    shapes.append(image_slice.shape[-2:])
                    writer.send(image_slice)

            return dimwise_max(shapes), None if max_id is None else int(max_id)


T = TypeVar("T", bound=Tuple[int, ...])


def dimwise_max(vectors: Sequence[T]) -> T:
    if len(vectors) == 1:
        return vectors[0]
    else:
        return cast(T, tuple(map(max, *vectors)))
