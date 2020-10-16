from wkcuber.mag import Mag


class Resolution:
    def __init__(self, mag):
        self._mag = Mag(mag)

    def _to_json(self) -> dict:
        return {"resolution": self.mag.to_array()}

    @classmethod
    def _from_json(cls, json_data):
        return cls(json_data["resolution"])

    @property
    def mag(self) -> Mag:
        return self._mag


class WkResolution(Resolution):
    def __init__(self, mag, cube_length):
        super().__init__(mag)
        self._cube_length = cube_length

    def _to_json(self) -> dict:
        return {"resolution": self.mag.to_array(), "cubeLength": self.cube_length}

    @classmethod
    def _from_json(cls, json_data):
        return cls(json_data["resolution"], json_data["cubeLength"])

    @property
    def mag(self) -> Mag:
        return self._mag

    @property
    def cube_length(self) -> int:
        return self._cube_length
