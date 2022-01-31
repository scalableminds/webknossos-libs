from webknossos.administration import Project


def main() -> None:
    p = Project.get_by_name("sampleProject")
    tasks = p.get_tasks()
    print(tasks)
    annotations = [t.get_annotation_infos() for t in tasks]
    print(annotations)


if __name__ == "__main__":
    main()
