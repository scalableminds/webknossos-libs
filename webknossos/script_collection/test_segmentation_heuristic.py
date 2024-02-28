from pathlib import Path
from typing import List

import webknossos as wk
from webknossos.dataset._utils.segmentation_recognition import (
    THRESHOLD,
    guess_if_segmentation_from_view,
    sample_distinct_values_per_vx,
)


def main(path_to_datasets: Path) -> None:
    correct_prediction_count = 0
    wrong_prediction_count = 0
    wrongly_predicted: List[str] = []
    correctly_predicted: List[str] = []
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
                        wrong_prediction_count += 1
                        wrongly_predicted.append(
                            f"{dataset.name}/{color_layer.name}: guessed segmentation, but is color. Score = {score}."
                        )
                    else:
                        correct_prediction_count += 1
                        correctly_predicted.append(
                            f"{dataset.name}/{color_layer.name}: correct as color. Score = {score}."
                        )
                except Exception:
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
                        correct_prediction_count += 1
                        correctly_predicted.append(
                            f"{dataset.name}/{segmentation_layer.name}: correct as segmentation. Score = {score}."
                        )
                    else:
                        wrong_prediction_count += 1
                        wrongly_predicted.append(
                            f"{dataset.name}/{segmentation_layer.name}: guessed color, but is segmentation. Score = {score}."
                        )
                except Exception:
                    pass
        except Exception:
            pass
    print(
        f"{wrong_prediction_count} wrong, {correct_prediction_count} correct with threshold {THRESHOLD}."
    )
    for element in correctly_predicted:
        print(element)
    for element in wrongly_predicted:
        print(element)


if __name__ == "__main__":
    main(Path("../webknossos/binaryData/sample_organization/"))
