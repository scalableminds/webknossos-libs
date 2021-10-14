from typing_extensions import Literal


class LayerCategories:
    """
    There are two different types of layers.
    This class can be used to specify the type of a layer during creation:
    ```python
    from webknossos.dataset.dataset import Dataset

    dataset = Dataset(<path_to_dataset>)
    # Adds a new layer
    layer = dataset.add_layer("color", LayerCategories.COLOR_TYPE)
    ```
    """

    COLOR_TYPE: Literal["color"] = "color"
    SEGMENTATION_TYPE: Literal["segmentation"] = "segmentation"


# LayerCategoryType = Literal[LayerCategories.COLOR_TYPE, LayerCategories.SEGMENTATION_TYPE]
LayerCategoryType = Literal["color", "segmentation"]
