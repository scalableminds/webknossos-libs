import webknossos as wk
from webknossos.dataset.remote_folder import RemoteFolder


def main() -> None:
    # Get the folder id of the upload destination:
    folder_id = RemoteFolder.get_by_path("Datasets/").id

    # Announce the manual upload of a new dataset:
    wk.Dataset.announce_manual_upload(
        dataset_name="my_new_dataset_name",
        organization="sample_organization",
        initial_team_ids=[],
        folder_id=folder_id,
    )


if __name__ == "__main__":
    main()
