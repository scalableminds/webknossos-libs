from webknossos.task import Task


def main() -> None:
    task = Task.get_by_id("61f151c10100000a01249afe")
    print(f"Task: {task}")


if __name__ == "__main__":
    main()
