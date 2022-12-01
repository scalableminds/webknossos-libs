import warnings
from contextlib import contextmanager, nullcontext
from itertools import chain
from os import PathLike
from pathlib import Path
from typing import (
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)
from urllib.error import HTTPError

import numpy as np

try:
    from webknossos.dataset._utils.pims_czi_reader import PimsCziReader
except ImportError:
    PimsCziReader = type(None)  # type: ignore[misc,assignment]

from webknossos.dataset.layer import DTypeLike
from webknossos.dataset.mag_view import MagView
from webknossos.geometry.vec3_int import Vec3Int

try:
    import pims
except ImportError as import_error:
    raise RuntimeError(
        "Cannot import pims, please install it e.g. using 'webknossos[all]'"
    ) from import_error


# Fix ImageIOReader not handling channels correctly. This might get fixed via
# https://github.com/soft-matter/pims/pull/430
pims.ImageIOReader.frame_shape = pims.FramesSequenceND.frame_shape


def _assume_color_channel(dim_size: int, dtype: np.dtype) -> bool:
    return dim_size == 1 or (dim_size == 3 and dtype == np.dtype("uint8"))


class PimsImages:
    dtype: DTypeLike
    expected_shape: Vec3Int
    num_channels: int

    def __init__(
        self,
        images: Union[str, Path, "pims.FramesSequence", List[Union[str, PathLike]]],
        channel: Optional[int],
        timepoint: Optional[int],
        czi_channel: Optional[int],
        swap_xy: bool,
        flip_x: bool,
        flip_y: bool,
        flip_z: bool,
        use_bioformats: bool,
        is_segmentation: bool,
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
        self._channel = channel
        self._timepoint = timepoint
        self._czi_channel = czi_channel
        self._swap_xy = swap_xy
        self._flip_x = flip_x
        self._flip_y = flip_y
        self._flip_z = flip_z
        self._use_bioformats = use_bioformats

        ## attributes that will be set in __init__()
        self._iter_dim = None
        self._possible_layers = {}
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
                    # assuming channels come last. This might get fixed via
                    # https://github.com/soft-matter/pims/pull/430
                    if (
                        len(images._shape) >= len(images.sizes)
                        and "c" not in images.sizes
                    ):
                        images._init_axis("c", images._shape[-1])
                        self._init_c_axis = True

                if isinstance(images, PimsCziReader):
                    available_czi_channels = images.available_czi_channels()
                    if len(available_czi_channels) > 1:
                        self._possible_layers["czi_channel"] = available_czi_channels

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
                        if self._iter_dim == "":
                            self._iter_dim = "t"
                        else:
                            self._default_coords["t"] = 0
                            self._possible_layers["timepoint"] = list(
                                range(0, images.sizes["t"])
                            )
                    elif "t" in images.axes:
                        self._default_coords["t"] = 0
                else:
                    assert "t" in images.axes
                    self._default_coords["t"] = timepoint
            else:
                # Fallback for generic pims classes that do not name their
                # dimensions as pims.FramesSequenceND does:

                _allow_channels_first = not is_segmentation
                if isinstance(images, (pims.ImageSequence, pims.ReaderSequence)):
                    _allow_channels_first = False

                if len(images.shape) == 2:
                    # Assume yx
                    self._img_dims = "yx"
                    self._iter_dim = ""
                elif len(images.shape) == 3:
                    # Assume yxc, cyx or zyx
                    if _assume_color_channel(images.shape[2], images.dtype):
                        self._img_dims = "yxc"
                        self._iter_dim = ""
                    elif images.shape[0] == 1 or (
                        _allow_channels_first
                        and _assume_color_channel(images.shape[0], images.dtype)
                    ):
                        self._img_dims = "cyx"
                        self._iter_dim = ""
                    else:
                        self._img_dims = "yx"
                        self._iter_dim = "z"
                elif len(images.shape) == 4:
                    # Assume zcyx or zyxc
                    if images.shape[1] == 1 or _assume_color_channel(
                        images.shape[1], images.dtype
                    ):
                        self._img_dims = "cyx"
                    else:
                        self._img_dims = "yxc"
                    self._iter_dim = "z"
                elif len(images.shape) == 5:
                    # Assume tzcyx or tzyxc
                    # t has to be constant for this reader to obtain 4D image
                    # (only possible if not specified manually already, since
                    # the timepoint would already be indexed here and the
                    # 5th dimension would be something else)
                    if timepoint is not None:
                        raise RuntimeError(
                            f"Got {len(images.shape)} axes for the images after "
                            + "removing time dimension, can only map to 3D+channels."
                        )

                    if _assume_color_channel(images.shape[2], images.dtype):
                        self._img_dims = "cyx"
                    else:
                        self._img_dims = "yxc"
                    self._iter_dim = "z"
                    self._timepoint = 0
                    if images.shape[0] > 1:
                        self._possible_layers["timepoint"] = list(
                            range(0, images.shape[0])
                        )
                else:
                    raise RuntimeError(
                        f"Got {len(images.shape)} axes for the images, "
                        + "cannot map to 3D+channels+timepoints."
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
        if self._channel is not None:
            assert (
                self._channel < self.num_channels
            ), f"Selected channel {self._channel} (0-indexed), but only {self.num_channels} channels are available."
            self.num_channels = 1
        else:
            if self.num_channels == 2:
                self._possible_layers["channel"] = [0, 1]
                self.num_channels = 1
                self._channel = 0
            elif self.num_channels > 3:
                self._possible_layers["channel"] = list(range(0, self.num_channels))
                self.num_channels = 3
                self._first_n_channels = 3

    def _normalize_original_images(self) -> Union[str, List[str]]:
        original_images = self._original_images
        if isinstance(original_images, (str, Path)):
            original_images_path = Path(original_images)
            if original_images_path.is_dir():
                valid_suffixes = get_valid_pims_suffixes()
                original_images = [
                    str(i)
                    for i in original_images_path.glob("**/*")
                    if i.is_file() and i.suffix.lstrip(".") in valid_suffixes
                ]
                if len(original_images) == 1:
                    original_images = original_images[0]
        if isinstance(original_images, str):
            return original_images
        elif isinstance(original_images, Iterable):
            return [str(i) for i in original_images]
        else:
            return str(original_images)

    def _open_bioformats_images_raw(
        self,
        original_images: Union[str, List[str]],
        previous_exceptions: List[Exception],
    ) -> pims.FramesSequence:
        try:
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

            if "*" in str(original_images) or isinstance(original_images, list):
                images_context_manager = pims.ReaderSequence(
                    original_images, pims.bioformats.BioformatsReader
                )
            else:
                images_context_manager = pims.bioformats.BioformatsReader(
                    original_images
                )
        except Exception as e:
            if len(previous_exceptions) == 0:
                raise e
            else:
                previous_exceptions.append(e)
                previous_exceptions_str = "\n".join(
                    f"{type(e).__name__}: {str(e)}" for e in previous_exceptions
                )
                raise ValueError(
                    f"Tried to open the images {self._original_images} with different methods, "
                    + f"none succeded. The following errors were raised:\n{previous_exceptions_str}"
                ) from None
        return images_context_manager

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
                pims_open_exceptions = []
                original_images = self._normalize_original_images()
                if not self._use_bioformats:
                    try:
                        open_kwargs = {}
                        if self._czi_channel is not None:
                            open_kwargs["czi_channel"] = self._czi_channel
                        images_context_manager = pims.open(original_images)
                    except Exception as e1:
                        pims_open_exceptions.append(e1)
                        try:
                            images_context_manager = pims.ImageSequence(original_images)
                        except Exception as e2:
                            pims_open_exceptions.append(e2)
                            self._use_bioformats = True

                if self._use_bioformats:
                    images_context_manager = self._open_bioformats_images_raw(
                        original_images, pims_open_exceptions
                    )

            with images_context_manager as images:
                if isinstance(images, pims.FramesSequenceND):
                    if hasattr(self, "_img_dims"):
                        # first part of __init__() has happened
                        images.default_coords.update(self._default_coords)
                        if self._init_c_axis and "c" not in images.sizes:
                            # Bugfix for ImageIOReader which misses channel axis sometimes,
                            # assuming channels come last. _init_c_axis is set in __init__().
                            # This might get fixed via https://github.com/soft-matter/pims/pull/430
                            images._init_axis("c", images._shape[-1])
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
        self,
        args: Tuple[int, int],
        mag_view: MagView,
        is_segmentation: bool,
        dtype: Optional[DTypeLike] = None,
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
                relative_offset=(0, 0, z_start * mag_view.mag.z),
                buffer_size=mag_view.info.chunk_shape.z,
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
                        if self._channel is not None:
                            image_slice = image_slice[self._channel : self._channel + 1]
                        elif self._first_n_channels is not None:
                            image_slice = image_slice[: self._first_n_channels]
                        assert image_slice.shape[0] == self.num_channels, (
                            f"Image shape {image_slice.shape} does not fit to the number of channels "
                            + f"{self.num_channels} which are expected in the first axis."
                        )

                    if self._flip_x:
                        image_slice = np.flip(image_slice, -2)
                    if self._flip_y:
                        image_slice = np.flip(image_slice, -1)

                    if dtype is not None:
                        image_slice = image_slice.astype(dtype, order="F")

                    if max_id is not None:
                        max_id = max(max_id, image_slice.max())
                    shapes.append(image_slice.shape[-2:])
                    writer.send(image_slice)

            return dimwise_max(shapes), None if max_id is None else int(max_id)

    def get_possible_layers(self) -> Optional[Dict["str", List[int]]]:
        if len(self._possible_layers) == 0:
            return None
        else:
            return self._possible_layers


T = TypeVar("T", bound=Tuple[int, ...])


def dimwise_max(vectors: Sequence[T]) -> T:
    if len(vectors) == 1:
        return vectors[0]
    else:
        return cast(T, tuple(map(max, *vectors)))


C = TypeVar("C", bound=Type)


def _recursive_subclasses(cls: C) -> List[C]:
    "Return all subclasses (and their subclasses, etc.)."
    # Source: http://stackoverflow.com/a/3862957/1221924
    return cls.__subclasses__() + [
        g for s in cls.__subclasses__() for g in _recursive_subclasses(s)
    ]


def _get_all_pims_handlers() -> Iterable[
    Type[Union[pims.FramesSequence, pims.FramesSequenceND]]
]:
    return chain(
        _recursive_subclasses(pims.FramesSequence),
        _recursive_subclasses(pims.FramesSequenceND),
    )


def get_valid_pims_suffixes() -> Set[str]:
    valid_suffixes = set()
    for pims_handler in _get_all_pims_handlers():
        valid_suffixes.update(pims_handler.class_exts())
    return valid_suffixes


def has_image_z_dimension(
    filepath: Path,
    use_bioformats: bool,
    is_segmentation: bool,
) -> bool:

    pims_images = PimsImages(
        filepath,
        use_bioformats=use_bioformats,
        is_segmentation=is_segmentation,
        # the following arguments shouldn't matter much for the Dataset.from_images method:
        channel=None,
        timepoint=None,
        czi_channel=None,
        swap_xy=False,
        flip_x=False,
        flip_y=False,
        flip_z=False,
    )

    return pims_images.expected_shape.z > 1
