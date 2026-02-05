import webknossos as wk


def main() -> None:
    # Explore a zarr dataset with webknossos by adding it as a remote dataset
    wk.RemoteDataset.explore_and_add_remote(
        "https://data-humerus.webknossos.org/data/zarr/b2275d664e4c2a96/HuaLab-CBA_Ca-mouse-unexposed-M1/color",
        "Ca-mouse-unexposed-M1",
        folder=wk.RemoteFolder.get_by_path("Datasets"),
    )


if __name__ == "__main__":
    main()
