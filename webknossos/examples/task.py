from webknossos.administration import Project, Task
from webknossos.annotation import AnnotationState


def main() -> None:
    p = Project.get_by_name("sampleProject")
    tasks = p.get_tasks()

    annotation_infos = [t.get_annotation_infos() for t in tasks]
    annotation_infos_flat = [a for infos in annotation_infos for a in infos]
    finished_annotation_infos = [
        a for a in annotation_infos_flat if a.state == AnnotationState.FINISHED
    ]

    finished_annotations = [a.download_annotation() for a in finished_annotation_infos]

    assert len(finished_annotations) > 0, "No annotations are finished yet!"

    task_type_id = "61f90e4efe0100b102553009"  # from wk web interface
    tasks = Task.create_from_annotations(
        task_type_id,
        "sampleExp",
        1,
        1,
        "sampleProject",
        None,
        None,
        finished_annotations,
    )
    print(f"New tasks: {tasks}")


if __name__ == "__main__":
    main()
