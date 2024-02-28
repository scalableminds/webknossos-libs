from pathlib import Path
from typing import List

import webknossos as wk
from webknossos.dataset._utils.segmentation_recognition import (
    THRESHOLD,
    guess_if_segmentation_from_view,
    sample_distinct_values_per_vx,
)


def main(path_to_datasets: Path) -> None:
    wrongly_predicted: List[str] = []
    correctly_predicted: List[str] = []
    failures: List[str] = []
    for path in [dir for dir in path_to_datasets.iterdir() if dir.is_dir()]:
        try:
            dataset = wk.Dataset.open(path)
            for color_layer in dataset.get_color_layers():
                try:
                    score = sample_distinct_values_per_vx(color_layer.get_finest_mag())
                    if (
                        guess_if_segmentation_from_view(color_layer.get_finest_mag())
                        == True
                    ):
                        wrongly_predicted.append(
                            f"{color_layer.path} guessed segmentation, but is color. Score = {score}."
                        )
                    else:
                        correctly_predicted.append(
                            f"{color_layer.path}: correct as color. Score = {score}."
                        )
                except Exception as e:
                    failures.append(f"Failed to analyze {color_layer.path}: {e}")
                    pass
            for segmentation_layer in dataset.get_segmentation_layers():
                try:
                    score = sample_distinct_values_per_vx(
                        segmentation_layer.get_finest_mag()
                    )
                    if (
                        guess_if_segmentation_from_view(
                            segmentation_layer.get_finest_mag()
                        )
                        == True
                    ):
                        correctly_predicted.append(
                            f"{segmentation_layer.path} correct as segmentation. Score = {score}."
                        )
                    else:
                        wrongly_predicted.append(
                            f"{segmentation_layer.path} guessed color, but is segmentation. Score = {score}."
                        )
                except Exception as e:
                    failures.append(f"Failed to analyze {segmentation_layer.path}: {e}")
        except Exception as e:
            failures.append(f"Failed to analyze dataset at {path}: {e}")
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
