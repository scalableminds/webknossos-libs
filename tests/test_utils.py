import numpy as np
from wkcuber.utils import get_chunks, get_regular_chunks

BLOCK_LEN = 32


def test_get_chunks():
    source = list(range(0, 48))
    target = list(get_chunks(source, 8))

    assert len(target) == 6
    assert target[0] == list(range(0, 8))


def test_get_regular_chunks():
    target = list(get_regular_chunks(4, 44, 8))

    assert len(target) == 6
    assert list(target[0]) == list(range(0, 8))
    assert list(target[-1]) == list(range(40, 48))
