from pathlib import Path
from typing import List

import webknossos as wk
from webknossos.dataset._utils.segmentation_recognition import (
    guess_if_segmentation_from_view,
)


def main(path_to_datasets: Path) -> None:
    correct_prediction = 0
    wrong_prediction = 0
    wrongly_predicted: List[str] = []
    for path in [dir for dir in path_to_datasets.iterdir() if dir.is_dir()]:
        try:
            dataset = wk.Dataset.open(path)
            for color_layer in dataset.get_color_layers():
                try:
                    if (
                        guess_if_segmentation_from_view(color_layer.get_finest_mag())
                        == True
                    ):
                        wrong_prediction += 1
                        wrongly_predicted += f"Dataset: {dataset.name} Layer: {color_layer.name} predicted as segmentation but is stored as color."
                    else:
                        correct_prediction += 1
                except Exception:
                    pass
            for segmentation_layer in dataset.get_segmentation_layers():
                try:
                    if (
                        guess_if_segmentation_from_view(
                            segmentation_layer.get_finest_mag()
                        )
                        == True
                    ):
                        correct_prediction += 1
                    else:
                        wrong_prediction += 1
                        wrongly_predicted += f"Dataset: {dataset.name} Layer: {segmentation_layer.name} predicted as color but is stored as segmentation."
                except Exception:
                    pass
        except Exception:
            pass
        print(
            f"Wrong predictions: {wrong_prediction}, Correct predictions: {correct_prediction}"
        )
    for element in wrongly_predicted:
        print(element)


if __name__ == "__main__":
    main(Path("../webknossos/binaryData/sample_organization/"))
