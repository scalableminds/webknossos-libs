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

    # Move/Copy the dataset manually to the datastore

    # After a few minutes the dataset list is updated
    # automatically. The dataset is then visible for all
    # users that are part of one team defined in initial_team_ids.
    # To trigger a reload of the dataset properties manually, use:

    wk.Dataset.trigger_reload(
        dataset_name="my_new_dataset_name", organization="sample_organization"
    )


if __name__ == "__main__":
    main()
