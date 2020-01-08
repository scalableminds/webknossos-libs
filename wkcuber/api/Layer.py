from shutil import rmtree
from os import mkdir
from os.path import join
import logging
from wkw import wkw

from wkcuber.api.MagDataset import MagDataset, WKMagDataset, TiffMagDataset
from wkcuber.mag import Mag


class Layer:
    def __init__(self, name, dataset, dtype, num_channels):
        self.name = name
        self.dataset = dataset
        self.dtype = dtype
        self.num_channels = num_channels
        self.mags = {}

        full_path = join(dataset.path, name)
        try:
            mkdir(full_path)
        except OSError:
            logging.info("Creation of Layer {} failed".format(full_path))

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
        self.dataset.properties.delete_mag(self.name, mag)
        # delete files on disk
        full_path = join(self.dataset.path, self.name, mag)
        rmtree(full_path)

    def __create_dir_for_mag__(self, mag):
        full_path = join(self.dataset.path, self.name, mag)

        try:
            mkdir(full_path)
        except OSError:
            logging.info("Creation of MagDataset {} failed".format(full_path))

    def __assert_mag_does_not_exist_yet__(self, mag):
        if mag in self.mags.keys():
            raise IndexError(
                "Adding mag {} failed. There is already a mag with this name".format(
                    mag
                )
            )


class WKLayer(Layer):
    def add_mag(self, mag, block_len=32, file_len=32, block_type=1):
        # normalize the name of the mag
        mag = Mag(mag).to_layer_name()

        self.__assert_mag_does_not_exist_yet__(mag)
        self.__create_dir_for_mag__(mag)

        self.mags[mag] = WKMagDataset.create(self, mag, block_len, file_len, block_type)
        self.dataset.properties.add_mag(self.name, mag, block_len * file_len)

        return self.mags[mag]

    def get_or_add_mag(self, mag, block_len=32, file_len=32, block_type=1):
        # normalize the name of the mag
        mag = Mag(mag).to_layer_name()

        if mag in self.mags.keys():
            assert self.mags[mag].header.block_len == block_len, (
                "Cannot get_or_add_mag: The mag %s already exists, but the block lengths do not match"
                % mag
            )
            assert self.mags[mag].header.file_len == file_len, (
                "Cannot get_or_add_mag: The layer %s already exists, but the file lengths do not match"
                % mag
            )
            assert self.mags[mag].header.block_type == block_type, (
                "Cannot get_or_add_mag: The layer %s already exists, but the block types do not match"
                % mag
            )
            return self.get_mag(mag)
        else:
            return self.add_mag(mag, block_len, file_len, block_type)

    def setup_mag(self, mag):
        # This method is used to initialize the mag when opening the Dataset. This does not create e.g. folders.

        # normalize the name of the mag
        mag = Mag(mag).to_layer_name()

        self.__assert_mag_does_not_exist_yet__(mag)

        wk_header = wkw.Dataset.open(join(self.dataset.path, self.name, mag)).header

        self.mags[mag] = WKMagDataset(
            self, mag, wk_header.block_len, wk_header.file_len, wk_header.block_type
        )
        self.dataset.properties.add_mag(
            self.name, mag, wk_header.block_len * wk_header.file_len
        )


class TiffLayer(Layer):
    def add_mag(self, mag):
        # normalize the name of the mag
        mag = Mag(mag).to_layer_name()

        self.__assert_mag_does_not_exist_yet__(mag)
        self.__create_dir_for_mag__(mag)

        self.mags[mag] = TiffMagDataset.create(self, mag, self.dataset.properties.pattern)
        self.dataset.properties.add_mag(self.name, mag)

        return self.mags[mag]

    def get_or_add_mag(self, mag):
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

        self.__assert_mag_does_not_exist_yet__(mag)

        self.mags[mag] = TiffMagDataset(self, mag, self.dataset.properties.pattern)
        self.dataset.properties.add_mag(self.name, mag)
