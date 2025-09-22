from enum import Enum


class DataFormat(Enum):
    WKW = "wkw"
    Zarr = "zarr"
    Zarr3 = "zarr3"

    def __str__(self) -> str:
        return self.value


class AttachmentDataFormat(Enum):
    Zarr3 = "zarr3"
    HDF5 = "hdf5"
    JSON = "json"

    def __str__(self) -> str:
        return self.value
