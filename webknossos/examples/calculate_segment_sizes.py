import numpy as np

import webknossos as wk

ANNOTATION_URL = (
    "https://webknossos.org/annotations/Explorational/622b130a010000220122ebd1"
)


def main() -> None:

    annotation = wk.Annotation.download(ANNOTATION_URL)
    stats_per_id = {}
    with annotation.temporary_volume_layer_copy() as seg_layer:
        scale = np.array(annotation.scale) * seg_layer.get_best_mag().mag.to_np()
        seg_data = np.stack(
            [view.read() for view in seg_layer.get_best_mag().get_views_on_disk()]
        )
        uniques, counts = np.unique(seg_data, return_counts=True)
        for _id, count in zip(uniques, counts):
            if _id == 0:
                continue
            voxel_size_in_nm3 = scale[0] * scale[1] * scale[2]

            volume = count * voxel_size_in_nm3
            stats_per_id[_id] = (count, volume)

    for _id, (count, volume) in stats_per_id.items():
        print(f"ID={_id} Voxel Count={count} Volume={volume}nm³")


if __name__ == "__main__":
    main()
