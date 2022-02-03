from webknossos.administration import Project, Task
from webknossos.annotation import AnnotationState


def main() -> None:
    # Assume running annotation project with previously created tasks.

    # Fetch statistics about how many annotation tasks have been completed yet.
    p = Project.get_by_name("sampleProject")
    tasks = p.get_tasks()
    total_active_instances = sum([t.status.active_instance_count for t in tasks])
    total_open_instances = sum([t.status.open_instance_count for t in tasks])
    print(f"There are {total_active_instances} active and {total_open_instances} open task instances.")

    # Fetch info about annotation instances that annotators have already started
    annotation_infos = [t.get_annotation_infos() for t in tasks]
    annotation_infos_flat = [a for infos in annotation_infos for a in infos]

    # Filter out all that are not yet finished
    finished_annotation_infos = [
        a for a in annotation_infos_flat if a.state == AnnotationState.FINISHED
    ]

    # Download the content of the finished annotations
    finished_annotations = [a.download_annotation() for a in finished_annotation_infos]

    assert len(finished_annotations) > 0, "No annotations are finished yet!"

    # Assume a second task type is in place that instructs annotators to take the previously created
    # annotations and perform a secondary annotation step on them
    # (e.g. first task is to seed positions, second is to fully label around those)

    task_type_id = "61f90e4efe0100b102553009"  # from webKnossos web interface
    tasks = Task.create_from_annotations(
        task_type_id, # New task type instructs annotators what to do in the task
        "sampleExp", # Only annotators with the experience "sampleExp" of at least value 2 will get this task
        2, # experience value (see line above)
        1, # create one instance of each task, no redundant annotating
        "sampleProject2", # The new task is part of a second project to track the progress separately
        None, # No custom user script for this task
        None, # No restricted bounding box for this task
        finished_annotations, # create a new task based on each annotation that was previously finished
    )
    print(f"New tasks: {tasks}")


if __name__ == "__main__":
    main()
