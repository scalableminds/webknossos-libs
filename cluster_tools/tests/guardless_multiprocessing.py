import multiprocessing

from cluster_tools import get_executor

"""
This file is an example of an incorrect script setup.
The module does not use the
    if __name__ == "__main__":
      main()
pattern which can lead to bugs when using multiprocessing.
The clustertools will detect the wrong usage and emit a warning.
This file is used to test the warning mechanism.
"""


multiprocessing.set_start_method("spawn", force=True)


def worker_fn() -> bool:
    return True


def main() -> None:
    res_fut = get_executor("multiprocessing").submit(worker_fn)
    assert res_fut.result() == True, "Function should return True"
    print("success")


main()
