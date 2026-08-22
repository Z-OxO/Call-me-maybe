import os
import sys
from typing import Any, TextIO

from pydantic_core import ErrorDetails

from src.constants import Colors


def _paint(text: str, code: str, stream: TextIO | None = None) -> str:
    """Wrap text in an ANSI color, unless colors are disabled."""
    out = stream if stream is not None else sys.stderr
    if os.environ.get("NO_COLOR") or not out.isatty():
        return text
    return f"{code}{text}{Colors.RESET}"


def _subject(error: ErrorDetails) -> str:
    """Name the offending object, falling back to its index.

    Pydantic only exposes the whole object as ``input`` for some error
    types (``missing`` notably); elsewhere it holds the faulty leaf.
    """
    src = error["input"]
    if isinstance(src, dict) and isinstance(src.get("name"), str):
        return str(src["name"])
    loc = error["loc"]
    return f"Object {loc[0]}" if loc else "Catalog"


def _location(error: ErrorDetails) -> str:
    """Dotted path of the faulty field, relative to its object."""
    names = [str(p) for p in error["loc"] if not isinstance(p, int)]
    return ".".join(names) if names else "value"


def _get_formatted(error: ErrorDetails) -> str:
    """Render one pydantic error as a single readable line."""
    subject = _subject(error)
    path = _paint(_location(error), Colors.CYAN)
    ctx: dict[str, Any] = error.get("ctx", {})

    match error["type"]:
        case "missing":
            return f"{subject}: {path} is required but missing"
        case "string_too_short":
            return f"{subject}: {path} cannot be empty"
        case "string_type":
            return f"{subject}: {path} must be text"
        case "dict_type" | "model_type":
            return f"{subject}: {path} must be an object"
        case "list_type":
            return f"{subject}: {path} must be an array"
        case "literal_error":
            expected = _paint(str(ctx.get("expected", "?")), Colors.GREEN)
            got = _paint(str(error["input"]), Colors.YELLOW)
            return f"{subject}: {path} got {got}, expected {expected}"
        case "duplicate_names":
            dup = _paint(", ".join(ctx.get("names", [])), Colors.CYAN)
            return f"Duplicate function names: {dup}"
        case "empty_catalog":
            return "Functions catalog cannot be empty"
        case "value_error":
            return str(ctx.get("error", error["msg"]))
        case "json_invalid":
            detail = str(error["msg"]).removeprefix("Invalid JSON: ")
            return f"File is not valid JSON \u2014 {detail}"
        case _:
            return f"{subject}: {path}: {error['msg']}"


def print_formatted_errors(errors: list[ErrorDetails]) -> None:
    """Print every validation error, one per line, on stderr."""
    for error in errors:
        print(_get_formatted(error), file=sys.stderr)
