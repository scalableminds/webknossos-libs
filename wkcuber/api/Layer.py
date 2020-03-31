from shutil import rmtree
from os.path import join
from os import makedirs
from wkw import wkw

from wkcuber.api.MagDataset import (
    MagDataset,
    WKMagDataset,
    TiffMagDataset,
    TiledTiffMagDataset,
)
from wkcuber.mag import Mag


class Layer:

    COLOR_TYPE = "color"
    SEGMENTATION_TYPE = "segmentation"

    def __init__(self, name, dataset, dtype, num_channels):
        self.name = name
        self.dataset = dataset
        self.dtype = dtype
        self.num_channels = num_channels
        self.mags = {}

        full_path = join(dataset.path, name)
        makedirs(full_path, exist_ok=True)

    def get_mag(self, mag) -> MagDataset:
        if mag not in self.mags.keys():
            raise IndexError("The mag {} is not a mag of this layer".format(mag))
        return self.mags[mag]

    def delete_mag(self, mag):
        if mag not in self.mags.keys():
            raise IndexError(
                "Deleting mag {} failed. There is no mag with this name".format(mag)
            )

        del self.mags[mag]
        self.dataset.properties._delete_mag(self.name, mag)
        # delete files on disk
        full_path = join(self.dataset.path, self.name, mag)
        rmtree(full_path)

    def _create_dir_for_mag(self, mag):
        full_path = join(self.dataset.path, self.name, mag)
        makedirs(full_path, exist_ok=True)

    def _assert_mag_does_not_exist_yet(self, mag):
        if mag in self.mags.keys():
            raise IndexError(
                "Adding mag {} failed. There is already a mag with this name".format(
                    mag
                )
            )


class WKLayer(Layer):
    def add_mag(
        self, mag, block_len=None, file_len=None, block_type=None
    ) -> WKMagDataset:
        if block_len is None:
            block_len = 32
        if file_len is None:
            file_len = 32
        if block_type is None:
            block_type = wkw.Header.BLOCK_TYPE_RAW

        # normalize the name of the mag
        mag = Mag(mag).to_layer_name()

        self._assert_mag_does_not_exist_yet(mag)
        self._create_dir_for_mag(mag)

        self.mags[mag] = WKMagDataset.create(self, mag, block_len, file_len, block_type)
        self.dataset.properties._add_mag(self.name, mag, block_len * file_len)

        return self.mags[mag]

    def get_or_add_mag(
        self, mag, block_len=None, file_len=None, block_type=None
    ) -> WKMagDataset:
        # normalize the name of the mag
        mag = Mag(mag).to_layer_name()

        if mag in self.mags.keys():
            assert (
                block_len is None or self.mags[mag].header.block_len == block_len
            ), f"Cannot get_or_add_mag: The mag {mag} already exists, but the block lengths do not match"
            assert (
                file_len is None or self.mags[mag].header.file_len == file_len
            ), f"Cannot get_or_add_mag: The mag {mag} already exists, but the file lengths do not match"
            assert (
                block_type is None or self.mags[mag].header.block_type == block_type
            ), f"Cannot get_or_add_mag: The mag {mag} already exists, but the block types do not match"
            return self.get_mag(mag)
        else:
            return self.add_mag(mag, block_len, file_len, block_type)

    def setup_mag(self, mag):
        # This method is used to initialize the mag when opening the Dataset. This does not create e.g. the wk_header.

        # normalize the name of the mag
        mag = Mag(mag).to_layer_name()

        self._assert_mag_does_not_exist_yet(mag)

        with wkw.Dataset.open(join(self.dataset.path, self.name, mag)) as wkw_dataset:
            wk_header = wkw_dataset.header

        self.mags[mag] = WKMagDataset(
            self, mag, wk_header.block_len, wk_header.file_len, wk_header.block_type
        )
        self.dataset.properties._add_mag(
            self.name, mag, wk_header.block_len * wk_header.file_len
        )


class TiffLayer(Layer):
    def add_mag(self, mag) -> MagDataset:
        # normalize the name of the mag
        mag = Mag(mag).to_layer_name()

        self._assert_mag_does_not_exist_yet(mag)
        self._create_dir_for_mag(mag)

        self.mags[mag] = self._get_mag_dataset_class().create(
            self, mag, self.dataset.properties.pattern
        )
        self.dataset.properties._add_mag(self.name, mag)

        return self.mags[mag]

    def get_or_add_mag(self, mag) -> MagDataset:
        # normalize the name of the mag
        mag = Mag(mag).to_layer_name()

        if mag in self.mags.keys():
            return self.get_mag(mag)
        else:
            return self.add_mag(mag)

    def setup_mag(self, mag):
        # This method is used to initialize the mag when opening the Dataset. This does not create e.g. folders.

        # normalize the name of the mag
        mag = Mag(mag).to_layer_name()

        self._assert_mag_does_not_exist_yet(mag)

        self.mags[mag] = self._get_mag_dataset_class()(
            self, mag, self.dataset.properties.pattern
        )
        self.dataset.properties._add_mag(self.name, mag)

    def _get_mag_dataset_class(self):
        return TiffMagDataset


class TiledTiffLayer(TiffLayer):
    def _get_mag_dataset_class(self):
        return TiledTiffMagDataset
