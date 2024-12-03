import inspect
import os
import sys
import warnings
from contextlib import contextmanager, nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from typing import Any, ContextManager, Iterator, Optional, Tuple, Type

import numpy as np
import pytest
from scipy.spatial import cKDTree

import webknossos as wk


@contextmanager
def tmp_cwd() -> Iterator[None]:
    prev_cwd = os.getcwd()
    with TemporaryDirectory() as new_cwd:
        os.chdir(new_cwd)
        try:
            yield
        finally:
            os.chdir(prev_cwd)


def exec_main_and_get_vars(
    module: ModuleType,
    *var_names: str,
    raises: Optional[Type[Exception]] = None,
    warns: Optional[str] = None,
) -> Tuple[Any, ...]:
    source = inspect.getsource(module)
    global_statements = "\n".join(f"    global {var_name}" for var_name in var_names)
    def_main_needle = "def main() -> None:\n"
    assert (
        def_main_needle in source
    ), "main() function could not be found in module. Failed to convert local to global vars."
    new_source = source.replace(
        def_main_needle, "def main() -> None:\n" + global_statements + "\n"
    )
    code = compile(new_source, str(module.__file__), "exec")
    exec(code, module.__dict__)
    cm_raises: ContextManager[Any]
    cm_warns: ContextManager[Any]
    if raises is None:
        cm_raises = nullcontext()
    else:
        cm_raises = pytest.raises(raises)
    if warns is None:
        cm_warns = nullcontext()
    else:
        cm_warns = pytest.warns(UserWarning, match=warns)
    with cm_warns:
        with cm_raises:
            module.main()  # type: ignore[attr-defined]

    return tuple(module.__dict__[var_name] for var_name in var_names)


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="Test is flaky for python 3.9 and 3.10."
)
def test_dataset_usage() -> None:
    import examples.dataset_usage as example

    (
        data_in_mag1,
        data_in_mag1_subset,
        data_in_mag2,
        data_in_mag2_subset,
    ) = exec_main_and_get_vars(
        example,
        "data_in_mag1",
        "data_in_mag1_subset",
        "data_in_mag2",
        "data_in_mag2_subset",
    )

    assert data_in_mag1.shape == (3, 512, 512, 32)
    assert data_in_mag1_subset.shape == (3, 512, 512, 32)
    assert data_in_mag2.shape == (3, 256, 256, 16)
    assert data_in_mag2_subset.shape == (3, 256, 256, 16)


def test_create_dataset_from_images() -> None:
    with tmp_cwd():
        import examples.create_dataset_from_images as example

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module="webknossos")
            (dataset,) = exec_main_and_get_vars(example, "dataset")
        assert dataset.voxel_size == (11, 11, 11)
        assert len(dataset.layers) == 1
        assert dataset.get_layer("tiff").get_finest_mag().read().shape == (
            1,
            265,
            265,
            257,
        )
        assert dataset.get_layer("tiff").dtype_per_channel == "uint8"


def test_image_stack_to_dataset() -> None:
    with tmp_cwd():
        import examples.image_stack_to_dataset as example
        from webknossos.dataset import COLOR_CATEGORY

        (dataset,) = exec_main_and_get_vars(example, "dataset")
        assert len(dataset.layers) == 1
        assert dataset.get_layer("test").category == COLOR_CATEGORY
        assert dataset.get_layer("test").get_finest_mag().read().shape == (
            1,
            265,
            265,
            2,
        )
        assert dataset.get_layer("test").dtype_per_channel == "uint8"


@pytest.mark.use_proxay
def test_apply_merger_mode() -> None:
    import examples.apply_merger_mode as example

    with wk.webknossos_context("http://localhost:9000", os.environ["WK_TOKEN"]):
        (out_mag1,) = exec_main_and_get_vars(example, "out_mag1")
        assert (
            out_mag1.read(absolute_offset=(3457, 3323, 1204), size=(1, 1, 1))[
                0, 0, 0, 0
            ]
            != 6016
        )
        assert (
            out_mag1.read(absolute_offset=(3457, 3323, 1204), size=(1, 1, 1))[
                0, 0, 0, 0
            ]
            == 1229599
        )


def test_calculate_segment_sizes() -> None:
    import examples.calculate_segment_sizes as example

    (stats_per_id,) = exec_main_and_get_vars(example, "stats_per_id")

    assert len(stats_per_id) == 2
    count1, volume1 = stats_per_id[1]
    count2, volume2 = stats_per_id[2]

    assert count1 == 11296 and volume1 == 39959065.20141486
    assert count2 == 12704 and volume2 == 44939798.540967986


def test_skeleton_synapse_candidates() -> None:
    import examples.skeleton_synapse_candidates as example

    synapse_parent_group, skeleton = exec_main_and_get_vars(
        example, "synapse_parent_group", "skeleton"
    )

    assert synapse_parent_group.get_total_node_count() == 91
    ids = [g.id for g in skeleton.flattened_trees()]
    id_set = set(ids)
    assert len(ids) == len(id_set), "Tree IDs are not unique."


@pytest.mark.use_proxay
def test_upload_image_data() -> None:
    with tmp_cwd():
        import examples.upload_image_data as example

        layer_nuclei, img, url = exec_main_and_get_vars(
            example, "layer_nuclei", "img", "url"
        )

        assert layer_nuclei.bounding_box.size == img.shape[1:]
        assert url.startswith("http://localhost:9000/datasets/Organization_X/cell_")


def test_download_image_data() -> None:
    with tmp_cwd():
        import examples.download_image_data as example

        (ds,) = exec_main_and_get_vars(example, "ds")

        assert list(ds.layers.keys()) == ["color"]


class _DummyNearestNeighborClassifier:
    """Faster replacement for a sklearn classifier,
    also removing the need for sklearn as a dependency."""

    labels: np.ndarray
    tree: cKDTree

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def fit(self, X: np.ndarray, labels: np.ndarray) -> None:
        MAX_FITTED_EXAMPLES = 64
        if X.shape[0] > MAX_FITTED_EXAMPLES:
            selection = np.random.default_rng(seed=42).choice(
                X.shape[0], MAX_FITTED_EXAMPLES, replace=False
            )
            self.labels = labels[selection]
            assert set(self.labels) == set(labels), (
                "Subsampling the examples omitted some labels, please use more examples.\n"
                + f"Currently MAX_FITTED_EXAMPLES is set to {MAX_FITTED_EXAMPLES}"
            )
            self.tree = cKDTree(X[selection])
        else:
            self.labels = labels
            self.tree = cKDTree(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        nearest_neighbors = self.tree.query(X)[1]
        return self.labels[nearest_neighbors]


@pytest.mark.use_proxay
def test_learned_segmenter() -> None:
    with tmp_cwd():
        from skimage.future import trainable_segmentation

        old_default_classifier = None
        if trainable_segmentation.has_sklearn:
            old_default_classifier = trainable_segmentation.RandomForestClassifier
        trainable_segmentation.RandomForestClassifier = _DummyNearestNeighborClassifier
        trainable_segmentation.has_sklearn = True
        import examples.learned_segmenter as example

        segmentation, url = exec_main_and_get_vars(example, "segmentation", "url")

        ids, id_counts = np.unique(segmentation, return_counts=True)
        counts = dict(zip(ids, id_counts))
        assert counts == {1: 209066, 2: 37803, 3: 164553, 4: 817378}
        assert (
            url
            == "http://localhost:9000/datasets/Organization_X/Skin_Layers_Dermis_and_Epidermis_segmented"
        )

        if old_default_classifier is None:
            del trainable_segmentation.RandomForestClassifier
            trainable_segmentation.has_sklearn = False
        else:
            trainable_segmentation.RandomForestClassifier = old_default_classifier


@pytest.mark.use_proxay
def test_user_times() -> None:
    import examples.user_times as example

    (df,) = exec_main_and_get_vars(example, "df")

    assert len(df) > 0
    assert sum(df.loc[:, (2021, 5)]) > 11
    assert "user_A@scalableminds.com" in df.index


@pytest.mark.use_proxay
def test_remote_datasets() -> None:
    import examples.remote_datasets as example

    with wk.webknossos_context("http://localhost:9000", os.environ["WK_TOKEN"]):
        (own_remote_datasets,) = exec_main_and_get_vars(
            example,
            "own_remote_datasets",
        )

        ds = own_remote_datasets["l4_sample"]
        assert ds.url == "http://localhost:9000/datasets/Organization_X/l4_sample"
        ds.tags = ["demo"]
        assert ds in wk.Dataset.get_remote_datasets(tags=["demo"]).values()


@pytest.mark.skipif(
    sys.version_info <= (3, 9), reason="Dask only supports Python >= 3.9"
)
def test_zarr_and_dask() -> None:
    import examples.zarr_and_dask as example

    (mean_value,) = exec_main_and_get_vars(example, "mean_value")

    assert 123 < mean_value < 125


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="Test is flaky for python 3.9 and 3.10."
)
@pytest.mark.use_proxay
def test_upload_tiff_stack() -> None:
    import examples.upload_tiff_stack as example

    with tmp_cwd():
        (remote_dataset,) = exec_main_and_get_vars(example, "remote_dataset")

        assert remote_dataset.url.startswith(
            "http://localhost:9000/datasets/Organization_X/tiff_dataset"
        )


@pytest.mark.use_proxay
def test_upload_dicom_stack() -> None:
    import examples.upload_dicom_stack as example

    with tmp_cwd():
        (remote_dataset,) = exec_main_and_get_vars(example, "remote_dataset")

        assert remote_dataset.url.startswith(
            "http://localhost:9000/datasets/Organization_X/dicom_dataset"
        )


def test_download_segments() -> None:
    import examples.download_segments as example

    with tmp_cwd():
        output_path = Path("l4_sample_segments")
        output_path.mkdir()

        (mag_view,) = exec_main_and_get_vars(example, "mag_view")

        assert (
            len(list(output_path.iterdir()))
            == 2 * mag_view.layer.bounding_box.size.z / mag_view.mag.z
        )


def test_download_tiff_stack() -> None:
    import examples.download_tiff_stack as example

    with tmp_cwd():
        output_path = Path("l4_sample_tiff")
        output_path.mkdir()

        (mag_view,) = exec_main_and_get_vars(example, "mag_view")

        assert (
            len(list(output_path.iterdir()))
            == mag_view.bounding_box.size.z / mag_view.mag.z
        )


def test_skeleton_path_length() -> None:
    from examples.skeleton_path_length import calculate_path_length

    # Public skeleton annotation by MH Lab
    annotation_id = "https://webknossos.org/annotations/62b191ef010000e80033e7c0"
    token = "123"
    calculate_path_length(annotation_id, token)


def test_load_annotation_file() -> None:
    from examples.load_annotation_from_file import load_annotation

    annotation_file = Path(
        "./tests/example_files/l4dense_motta_et_al_demo_v2__explorational.zip"
    ).resolve()

    with tmp_cwd():
        load_annotation(annotation_file)


def test_upsample_skeleton() -> None:
    import examples.upsample_skeleton as example

    with tmp_cwd():
        (skeleton,) = exec_main_and_get_vars(example, "skeleton")

        assert skeleton.voxel_size == (16, 16, 35)
