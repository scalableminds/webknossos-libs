from typing import Tuple, Type

import cluster_tools


class TestClass:
    pass


def deref_fun_helper(obj: Tuple[Type[TestClass], TestClass, int, int]) -> None:
    clss, inst, one, two = obj
    assert one == 1
    assert two == 2
    assert isinstance(inst, clss)


def test_dereferencing_main() -> None:
    with cluster_tools.get_executor(
        "slurm", debug=True, job_resources={"mem": "10M"}
    ) as executor:
        fut = executor.submit(deref_fun_helper, (TestClass, TestClass(), 1, 2))
        fut.result()
        futs = executor.map_to_futures(
            deref_fun_helper, [(TestClass, TestClass(), 1, 2)]
        )
        futs[0].result()


if __name__ == "__main__":
    # Validate that slurm_executor.submit also works when being called from a __main__ module
    test_dereferencing_main()
