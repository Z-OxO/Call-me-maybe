import sys
import os
from pydantic_core import ErrorDetails
from typing import Any, TextIO
from srcs.constants import Colors


def _paint(text: str, code: str, stream: TextIO | None = None) -> str:
    out = stream if stream is not None else sys.stderr
    if os.environ.get("NO_COLOR") or not out.isatty():
        return text
    return f"{code}{text}{Colors.RESET}"


def _get_formatted(error: ErrorDetails) -> str:
    """Render one pydantic error as a single readable line."""
    index = error["loc"][0] if error["loc"] else ""
    names = [
        str(p) for p in error["loc"] if not isinstance(p, int) and p != ""
    ]
    path = _paint(".".join(names) if names else "value", Colors.CYAN)
    ctx: dict[str, Any] = error.get("ctx", {})
    match error["type"]:
        case "missing":
            return f"Object {index}: {path} is required but missing"
        case "string_too_short":
            return f"Object {index}: {path} cannot be empty"
        case "string_type":
            return f"Object {index}: {path} must be text"
        case "dict_type" | "model_type":
            return f"Object {repr(index)}: path must be an object"
        case "list_type":
            return f"Object {index}: {path} must be an array"
        case "literal_error":
            allowed = _paint(str(ctx.get("expected", "?")), Colors.GREEN)
            got = _paint(error["input"], Colors.YELLOW)
            return f"Object {index}: {path} got {got}, expected {allowed}"
        case "duplicate_names":
            names = ", ".join(ctx.get("names", []))
            return f"Duplicate function names: {_paint(names, Colors.CYAN)}"
        case "empty_catalog":
            return "Functions catalog cannot be empty"
        case "value_error":
            return str(ctx.get("error", error["msg"]))
        case "json_invalid":
            detail = str(error["msg"]).removeprefix("Invalid JSON: ")
            return f"file is not valid JSON \u2014 {detail}"
        case _:
            return f"{path}: {error['msg']}"


def print_formatted_errors(errors: list[ErrorDetails]):
    for error in errors:
        print(_get_formatted(error), file=sys.stderr)
