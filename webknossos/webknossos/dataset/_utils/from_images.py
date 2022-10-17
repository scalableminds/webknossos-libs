from pathlib import Path


def guess_if_segmentation_path(filepath: Path) -> bool:
    lowercase_filepath = str(filepath).lower()
    return any(i in lowercase_filepath for i in ["segmentation", "labels"])
