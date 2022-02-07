from webknossos import AnnotationState, Project, Task


def main() -> None:
    # Assume running annotation project with previously created tasks.
    # Compare webKnossos documentation at https://docs.webknossos.org/webknossos/tasks.html

    # Fetch statistics about how many annotation tasks have been completed yet.
    sample_project = Project.get_by_name("sampleProject")
    tasks = sample_project.get_tasks()
    total_active_instances = sum([task.status.active_instance_count for task in tasks])
    total_open_instances = sum([task.status.open_instance_count for task in tasks])
    print(
        f"There are {total_active_instances} active and {total_open_instances} open task instances."
    )

    # Find and download all of the project’s annotations that are already finished by annotators
    finished_annotations = []
    for task in tasks:
        for annotation_info in task.get_annotation_infos():
            if annotation_info.state == AnnotationState.FINISHED:
                finished_annotation = annotation_info.download_annotation()
                finished_annotations.append(finished_annotation)

    assert len(finished_annotations) > 0, "No annotations are finished yet!"

    # Assume a second task type is in place that instructs annotators to take the previously created
    # annotations and perform a secondary annotation step on them
    # (e.g. first task is to seed positions, second is to fully label around those)

    task_type_id = "61f90e4efe0100b102553009"  # from webKnossos web interface
    tasks = Task.create_from_annotations(
        task_type_id=task_type_id,  # New task type instructs annotators what to do in the task
        project_name="sampleProject2",
        base_annotations=finished_annotations,
        needed_experience_domain="sampleExp",  # Only annotators with the experience "sampleExp" of at least value 2 will get this task
        needed_experience_value=2,
    )
    print(f"New tasks: {tasks}")


if __name__ == "__main__":
    main()
