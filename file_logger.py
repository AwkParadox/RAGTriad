from pathlib import Path
from typing import Optional
import datetime


class FileLogger:
    """Simple append-only logger that keeps file and console output in sync."""

    def __init__(
        self,
        filepath: str,
        overwrite: bool = False,
        mirror_stdout: bool = True,
    ):
        target = Path(filepath)
        target.parent.mkdir(parents=True, exist_ok=True)

        if not overwrite and target.exists():
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            target = target.with_name(f"{target.stem}_{timestamp}{target.suffix}")

        self.path = target
        self.mirror_stdout = mirror_stdout

        if overwrite or not self.path.exists():
            self.path.write_text("", encoding="utf-8")

    def log(self, message: Optional[str] = ""):
        line = message if message is not None else ""
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(f"{line}\n")
        if self.mirror_stdout:
            print(line)

