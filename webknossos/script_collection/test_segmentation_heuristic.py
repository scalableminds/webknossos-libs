import sys
from pathlib import Path

import webknossos as wk
from webknossos.dataset._utils.segmentation_recognition import (
    THRESHOLD,
    guess_category_from_view,
    sample_distinct_values_per_vx,
)


def looks_like_remote_dataset(path: Path) -> bool:
    return len([item for item in path.iterdir() if item.is_dir()]) == 0


def main(path_to_datasets: Path) -> None:
    wrongly_predicted: list[str] = []
    correctly_predicted: list[str] = []
    failures: list[str] = []
    dataset_paths = [
        item
        for item in path_to_datasets.iterdir()
        if item.is_dir() and not item.name.startswith(".")
    ]
    for dataset_path in dataset_paths:
        print(".", end="")
        sys.stdout.flush()
        if looks_like_remote_dataset(dataset_path):
            continue
        try:
            dataset = wk.Dataset.open(dataset_path)
            for color_layer in dataset.get_color_layers():
                try:
                    score = sample_distinct_values_per_vx(color_layer.get_finest_mag())
                    if (
                        guess_category_from_view(color_layer.get_finest_mag())
                        == "segmentation"
                    ):
                        wrongly_predicted.append(
                            f"{color_layer.path} guessed segmentation, but is color. Score = {score}."
                        )
                    else:
                        correctly_predicted.append(
                            f"{color_layer.path}: correct as color. Score = {score}."
                        )
                except Exception as e:  # pylint: disable=bare-except # noqa: PERF203
                    failures.append(f"Failed to analyze {color_layer.path}: {e}")
            for segmentation_layer in dataset.get_segmentation_layers():
                try:
                    score = sample_distinct_values_per_vx(
                        segmentation_layer.get_finest_mag()
                    )
                    if (
                        guess_category_from_view(segmentation_layer.get_finest_mag())
                        == "segmentation"
                    ):
                        correctly_predicted.append(
                            f"{segmentation_layer.path} correct as segmentation. Score = {score}."
                        )
                    else:
                        wrongly_predicted.append(
                            f"{segmentation_layer.path} guessed color, but is segmentation. Score = {score}."
                        )
                except Exception as e:  # pylint: disable=bare-except # noqa: PERF203
                    failures.append(f"Failed to analyze {segmentation_layer.path}: {e}")
        except Exception as e:  # pylint: disable=bare-except
            failures.append(f"Failed to analyze dataset at {dataset_path}: {e}")
    for element in correctly_predicted:
        print(element)
    print("")
    for element in wrongly_predicted:
        print(element)
    for element in failures:
        print(element)
    print("")
    print(
        f"\n{len(wrongly_predicted)} wrong, {len(correctly_predicted)} correct, {len(failures)} failures. Threshold {THRESHOLD}.\n"
    )


if __name__ == "__main__":
    main(Path("../webknossos/binaryData/sample_organization/"))
