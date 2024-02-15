import warnings
from contextlib import contextmanager, nullcontext
from functools import cached_property
from itertools import chain
from os import PathLike
from pathlib import Path
from typing import (
    ContextManager,
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
from natsort import natsorted
from numpy.typing import DTypeLike

from webknossos.geometry.bounding_box import BoundingBox
from webknossos.geometry.nd_bounding_box import NDBoundingBox

# pylint: disable=unused-import
try:
    from .pims_czi_reader import PimsCziReader
except ImportError:
    PimsCziReader = type(None)  # type: ignore[misc,assignment]

try:
    from .pims_dm_readers import (  # pylint: disable=unused-import
        PimsDm3Reader,
        PimsDm4Reader,
    )
except ImportError:
    pass

try:
    from .pims_imagej_tiff_reader import (  # pylint: disable=unused-import
        PimsImagejTiffReader,
    )
except ImportError:
    pass
# pylint: enable=unused-import

from ...geometry.vec_int import VecInt
from ..mag_view import MagView

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
        use_bioformats: Optional[bool],
        is_segmentation: bool,
    ) -> None:
        """
        During initialization the pims objects are examined and configured to produce
        ndarrays that follow the following form:
        (self._iter_axis, *self._bundle_axis)
        self._iter_axis can be a list of different axis or an empty list if the image is 2D.
        In the latter case, the inner 2D image is still wrapped in a single-element list
        by _open_images() to be consistent with 3D images.
        self._bundle_axis can consist of "x", "y" and "c", where "c" is optional and must be
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
        # _bundle_axes
        self._iter_axes = None
        self._iter_loop_size = None
        self._possible_layers = {}

        ## attributes only for pims.FramesSequenceND instances:
        # _default_coords
        # _init_c_axis

        ## attributes that will also be set in __init__()
        # dtype
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

                # An image slice should always consist of a 2D image. If there are multiple channels
                # the data of each channel should be part delivered too. Other
                if images.sizes.get("c", 1) > 1:
                    self._bundle_axes = ["c", "y", "x"]
                else:
                    if "c" in images.axes:
                        # When c-axis is not in _bundle_axes and _iter_axes its value at coordinate 0
                        # should be returned
                        self._default_coords["c"] = 0
                    self._bundle_axes = ["y", "x"]

                # All other axes are used to iterate over them. The last one is iterated the fastest.
                self._iter_axes = list(
                    set(images.axes).difference({*self._bundle_axes, "c", "z"})
                )
                if "z" in images.axes:
                    self._iter_axes.append("z")

                if self._timepoint is not None:
                    # if a timepoint is given, PimsImages should only generate image slices for that timepoint
                    if "t" in self._iter_axes:
                        self._iter_axes.remove("t")
                        self._default_coords["t"] = self._timepoint

                if len(self._iter_axes) > 1:
                    iter_size = 1
                    self._iter_loop_size = dict()
                    for axis, other_axis in zip(
                        self._iter_axes[-1:0:-1], self._iter_axes[-2::-1]
                    ):
                        self._iter_loop_size[other_axis] = (
                            iter_size := iter_size * images.sizes[axis]
                        )

            else:
                # Fallback for generic pims classes that do not name their
                # dimensions as pims.FramesSequenceND does:

                _allow_channels_first = not is_segmentation
                if isinstance(images, (pims.ImageSequence, pims.ReaderSequence)):
                    _allow_channels_first = False

                if len(images.shape) == 2:
                    # Assume yx
                    self._bundle_axes = ["y", "x"]
                    self._iter_axes = [""]
                elif len(images.shape) == 3:
                    # Assume yxc, cyx or zyx
                    if _assume_color_channel(images.shape[2], images.dtype):
                        self._bundle_axes = ["y", "x", "c"]
                        self._iter_axes = [""]
                    elif images.shape[0] == 1 or (
                        _allow_channels_first
                        and _assume_color_channel(images.shape[0], images.dtype)
                    ):
                        self._bundle_axes = ["c", "y", "x"]
                        self._iter_axes = [""]
                    else:
                        self._bundle_axes = ["y", "x"]
                        self._iter_axes = ["z"]
                elif len(images.shape) == 4:
                    # Assume zcyx or zyxc
                    if images.shape[1] == 1 or _assume_color_channel(
                        images.shape[1], images.dtype
                    ):
                        self._bundle_axes = ["c", "y", "x"]
                    else:
                        self._bundle_axes = ["y", "x", "c"]
                    self._iter_axes = ["z"]
                else:
                    raise RuntimeError(
                        f"Got {len(images.shape)} axes for the images, "
                        + "but don't have axes information. Try to open "
                        + "a N-dimensional image file with use_bioformats="
                        + "True."
                    )

        #########################
        # IDENTIFY NUM_CHANNELS #
        #########################

        with self._open_images() as images:
            try:
                c_index = self._bundle_axes.index("c")
                if isinstance(images, list):
                    images_shape = (len(images),) + cast(
                        pims.FramesSequence, images[0]
                    ).shape
                else:
                    images_shape = images.shape  # pylint: disable=no-member

                self.num_channels = images_shape[c_index + 1]

            except ValueError:
                self.num_channels = 1

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
                original_images = natsorted(
                    str(i)
                    for i in original_images_path.glob("**/*")
                    if i.is_file() and i.suffix.lstrip(".") in valid_suffixes
                )
                if len(original_images) == 1:
                    original_images = original_images[0]
        if isinstance(original_images, str):
            return original_images
        elif isinstance(original_images, Iterable):
            return [str(i) for i in original_images]
        else:
            return str(original_images)

    def _ensure_correct_bioformats_usage(
        self, images_context_manager: pims.FramesSequence
    ) -> None:
        if (
            isinstance(images_context_manager, pims.bioformats.BioformatsReader)
            and self._use_bioformats == False
        ):  # None is allowed
            raise RuntimeError(
                "Selected bioformats reader, but using bioformats is not allowed "
                + "(use_bioformats is False)."
            )

    def _try_open_pims_images(
        self, original_images: Union[str, List[str]], exceptions: List[Exception]
    ) -> Optional[pims.FramesSequence]:
        if self._use_bioformats:
            return None

        open_kwargs = {}
        if self._czi_channel is not None:
            open_kwargs["czi_channel"] = self._czi_channel

        # try normal pims.open
        def strategy_0() -> pims.FramesSequence:
            result = pims.open(original_images, **open_kwargs)
            self._ensure_correct_bioformats_usage(original_images)
            return result

        # try pims.ImageSequence, which uses skimage internally but works for multiple images
        strategy_1 = lambda: pims.ImageSequence(original_images)

        # for image lists, try to guess the correct reader using only the first image,
        # and apply that for all images via pims.ReaderSequence
        def strategy_2() -> pims.FramesSequence:
            if isinstance(original_images, list):
                # assuming the same reader works for all images:
                first_image_handler = pims.open(original_images[0], **open_kwargs)
                self._ensure_correct_bioformats_usage(first_image_handler)
                return pims.ReaderSequence(
                    original_images, type(first_image_handler), **open_kwargs
                )
            else:
                return None

        for strategy in [strategy_0, strategy_1, strategy_2]:
            try:
                images_context_manager = strategy()
            except Exception as e:
                exceptions.append(e)
            else:
                if images_context_manager is not None:
                    return images_context_manager
        return None

    def _try_open_bioformats_images_raw(
        self,
        original_images: Union[str, List[str]],
        exceptions: List[Exception],
    ) -> pims.FramesSequence:
        try:
            if self._use_bioformats == False:  # None is allowed
                raise RuntimeError(
                    "Using bioformats is not allowed (use_bioformats is False)."
                )

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

                # This is also part of the worker dockerfile to cache the
                # jar in the image, please update Dockerfile.worker in the
                # voxelytics repo accordingly when editing this.
                pims.bioformats.download_jar(version="6.7.0")

            if "*" in str(original_images) or isinstance(original_images, list):
                return pims.ReaderSequence(
                    original_images, pims.bioformats.BioformatsReader
                )
            else:
                return pims.bioformats.BioformatsReader(original_images)
        except Exception as e:
            exceptions.append(e)

    @contextmanager
    def _open_images(
        self,
    ) -> Iterator[Union[pims.FramesSequence, List[pims.FramesSequence]]]:
        """
        This yields well-defined images of the form (self._iter_axes, *self._bundle_axes),
        after IDENTIFY AXIS ORDER of __init__() has run.
        For a 2D image this is achieved by wrapping it in a list.
        """
        images_context_manager: Optional[ContextManager]
        with warnings.catch_warnings():
            if isinstance(self._original_images, pims.FramesSequenceND):
                images_context_manager = nullcontext(enter_result=self._original_images)
            else:
                exceptions: List[Exception] = []
                original_images = self._normalize_original_images()
                images_context_manager = None

                images_context_manager = self._try_open_pims_images(
                    original_images, exceptions
                )

                if images_context_manager is None:
                    images_context_manager = self._try_open_bioformats_images_raw(
                        original_images, exceptions
                    )

                if images_context_manager is None:
                    if len(exceptions) == 1:
                        raise exceptions[0]
                    else:
                        exceptions_str = "\n".join(
                            f"{type(e).__name__}: {str(e)}" for e in exceptions
                        )
                        raise ValueError(
                            f"Tried to open the images {self._original_images} with different methods, "
                            + f"none succeded. The following errors were raised:\n{exceptions_str}"
                        )

            with images_context_manager as images:
                if isinstance(images, pims.FramesSequenceND):
                    if hasattr(self, "_bundle_axes"):
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
                        images.bundle_axes = self._bundle_axes
                        images.iter_axes = self._iter_axes or []
                else:
                    if self._iter_axes == []:
                        # add outer list to wrap 2D images as 3D-like structure
                        images = [images]
                yield images

    def copy_to_view(
        self,
        args: Union[BoundingBox, NDBoundingBox],
        mag_view: MagView,
        is_segmentation: bool,
        dtype: Optional[DTypeLike] = None,
    ) -> Tuple[Tuple[int, int], Optional[int]]:
        """Copies the images according to the passed arguments to the given mag_view.
        args is expected to be a (ND)BoundingBox the start and end of the z-range, meant for usage with an executor.
        copy_to_view returns an iterable of image shapes and largest segment ids. When using this
        method a manual update of the bounding box and the largest segment id might be necessary.
        """
        relative_bbox = args

        assert relative_bbox.set_3d("size", (1, 1, 1)) == VecInt.ones(
            len(relative_bbox)
        ), "The delivered BoundingBox has to be flat except for x,y and z dimension."

        z_start, z_end = relative_bbox.get_bounds("z")
        shapes = []
        max_id: Optional[int]
        if is_segmentation:
            max_id = 0
        else:
            max_id = None

        with self._open_images() as images:
            if self._iter_axes is not None and self._iter_loop_size is not None:
                # select the range of images that represents one xyz combination
                lower_bounds = sum(
                    self._iter_loop_size[axis_name]
                    * relative_bbox.get_bounds(axis_name)[0]
                    for axis_name in self._iter_axes[:-1]
                )
                upper_bounds = lower_bounds + relative_bbox.get_shape("z")
                images = images[lower_bounds:upper_bounds]
            if self._flip_z:
                images = images[::-1]  # pylint: disable=unsubscriptable-object

            with mag_view.get_buffered_slice_writer(
                # Previously only z_start and its end were important, now the slice writer needs to know
                # which axis is currently written.
                relative_bounding_box=relative_bbox,
                buffer_size=mag_view.info.chunk_shape.z,
                # copy_to_view is typically used in a multiprocessing-context. Therefore the
                # buffered slice writer should not update the json file to avoid race conditions.
                json_update_allowed=False,
            ) as writer:
                for image_slice in images[z_start:z_end]:
                    image_slice = np.array(image_slice)
                    # place channels first
                    if "c" in self._bundle_axes:
                        image_slice = np.moveaxis(
                            image_slice,
                            source=self._bundle_axes.index("c"),
                            destination=0,
                        )
                    if "c" in self._bundle_axes:
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

                    if self._swap_xy is False:
                        image_slice = np.moveaxis(image_slice, -1, -2)

                    shapes.append(image_slice.shape[-2:])
                    writer.send(image_slice)

            return dimwise_max(shapes), None if max_id is None else int(max_id)

    def get_possible_layers(self) -> Optional[Dict["str", List[int]]]:
        if len(self._possible_layers) == 0:
            return None
        else:
            return self._possible_layers

    @cached_property
    def expected_bbox(self) -> NDBoundingBox:
        # replaces the previous expected_shape to enable n-dimensional input files
        with self._open_images() as images:
            if isinstance(images, list):
                images_shape = (len(images),) + cast(
                    pims.FramesSequence, images[0]
                ).shape
            else:
                images_shape = images.shape  # pylint: disable=no-member

            if self._iter_axes is None or len(self._iter_axes) <= 1:
                if self._bundle_axes.index("x") < self._bundle_axes.index("y"):
                    x_index, y_index = 1, 2
                else:
                    x_index, y_index = 2, 1
                if self._swap_xy:
                    x_index, y_index = y_index, x_index
                return BoundingBox(
                    (0, 0, 0),
                    (images_shape[x_index], images_shape[y_index], images_shape[0]),
                )
            else:
                if isinstance(images, pims.FramesSequenceND):
                    axes_names = self._iter_axes + [
                        axis for axis in self._bundle_axes if axis != "c"
                    ]
                    axes_sizes = [
                        images.sizes[axis]
                        for axis in axes_names  # pylint: disable=no-member
                    ]
                    axes_index = list(range(1, len(axes_names) + 1))
                    topleft = VecInt.zeros(len(axes_names))

                    if self._swap_xy:
                        x_index, y_index = axes_names.index("x"), axes_names.index("y")
                        axes_sizes[x_index], axes_sizes[y_index] = (
                            axes_sizes[y_index],
                            axes_sizes[x_index],
                        )

                    return NDBoundingBox(
                        topleft,
                        axes_sizes,
                        axes_names,
                        axes_index,
                    )

                raise ValueError(
                    "It seems as if you try to load an N-dimensional image from 2D images. This is currently not supported."
                )


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


def _get_all_pims_handlers() -> (
    Iterable[Type[Union[pims.FramesSequence, pims.FramesSequenceND]]]
):
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
    use_bioformats: Optional[bool],
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

    return pims_images.expected_bbox.get_shape("z") > 1
