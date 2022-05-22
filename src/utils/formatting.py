from typing import Optional

from rich import print as rich_print
from rich.console import Console
from rich.syntax import Syntax


# Color coded tracebacks
# install(show_locals=False, extra_lines=0)
console = Console()


# TODO: Allow for users to choose theme
def syntax_print(string: str, language: Optional[str] = "python", theme: Optional[str] = "monokai",
                 title: Optional[str] = None) -> None:
    if title is not None:
        console.rule(title)
    syntax = Syntax(string, language, theme=theme, line_numbers=True)
    console.print(syntax)


def pretty_print(*data):
    rich_print(*data)
