from enum import StrEnum
from pathlib import Path

FORBIDDEN_STR = frozenset('"') | {chr(c) for c in range(0x20)}
MAX_TOKEN = 67
DEFAULT_FUNCTIONS = Path("data/input/functions_definition.json")
DEFAULT_INPUT = Path("data/input/function_calling_tests.json")
DEFAULT_OUTPUT = Path("data/output/function_calling_results.json")


class Colors(StrEnum):
    RESET = "\033[0m"
    DIM = "\033[2m"
    YELLOW = "\033[33m"
    GREEN = "\033[32m"
    CYAN = "\033[36m"
