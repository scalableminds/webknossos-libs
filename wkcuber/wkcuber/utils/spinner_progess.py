"""WKcuber utils to enhance development experience."""

from rich.progress import Progress, SpinnerColumn, TextColumn


class SpinnerProgress(Progress):
    """A Progress subclass for spinner annimations with success and fail indicators."""

    def __init__(self):
        super().__init__(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
        )

    def __enter__(self) -> "SpinnerProgress":
        super().__enter__()
        return self

    def task_done(self, task_id) -> None:
        """Ends the spinning animation and displays a green checkmark before task description."""
        with self._lock:
            description = self._tasks[task_id].description
            self.update(
                task_id,
                completed=100,
                description=f"[green]\u2714[/green] {description} [green]Done.[/green]",
            )

    def task_failed(self, task_id) -> None:
        """Ends spinning animation and displays red cross."""

        with self._lock:
            description = self._tasks[task_id].description
            self.update(
                task_id,
                completed=100,
                description=f"[red]\u2718[/red] {description} [red]Failed.[/red]",
            )
