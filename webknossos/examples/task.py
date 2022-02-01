from webknossos.administration import Project, Task
from webknossos.annotation import Annotation


def main() -> None:
    p = Project.get_by_name("sampleProject")
    tasks = p.get_tasks()
    print(tasks)
    annotations = [t.get_annotation_infos() for t in tasks]
    print(annotations)

    a = Annotation.load("/home/f/scm/nml/file1.nml.zip")
    task_type_id = "61efb258f901004c02d637ee"  # from wk web interface
    tasks = Task.create_from_annotations(
        task_type_id, "sampleExp", 1, 1, "sampleProject", None, None, [a]
    )
    print(tasks)


if __name__ == "__main__":
    main()
