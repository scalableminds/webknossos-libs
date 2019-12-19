from shutil import rmtree
from os import mkdir
from os.path import join
import logging
import numpy as np

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

    def add_mag(
        self, mag, cube_length=1024
    ):  # cube_length is only important for WkwResolutions # TODO add additional Header parameter
        # normalize the name of the mag
        mag = Mag(mag).to_layer_name()

        if mag in self.mags.keys():
            raise IndexError(
                "Adding mag {} failed. There is already a mag with this name".format(
                    mag
                )
            )

        full_path = join(self.dataset.path, self.name, mag)

        try:
            mkdir(full_path)
        except OSError:
            logging.info("Creation of MagDataset {} failed".format(full_path))

        self.mags[mag] = self.__get_mag_type__().create(self, mag)
        self.dataset.properties.add_mag(self.name, mag, cube_length)

        return self.mags[mag]

    def setup_mag(self, mag, cube_length=1024):
        # This method is used to initialize the mag when opening the Dataset. This does not create e.g. folders.

        # normalize the name of the mag
        mag = Mag(mag).to_layer_name()

        if mag in self.mags.keys():
            raise IndexError(
                "Adding mag {} failed. There is already a mag with this name".format(
                    mag
                )
            )

        self.mags[mag] = self.__get_mag_type__()(self, mag)
        self.dataset.properties.add_mag(self.name, mag, cube_length)

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

    def __get_mag_type__(self):
        raise NotImplementedError


class WKLayer(Layer):
    def __get_mag_type__(self):
        return WKMagDataset


class TiffLayer(Layer):
    def __get_mag_type__(self):
        return TiffMagDataset
