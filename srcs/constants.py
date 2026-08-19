from enum import StrEnum


FORBIDDEN_STR = frozenset('"') | {chr(c) for c in range(0x20)}
MAX_TOKEN = 67


class Colors(StrEnum):
    RESET = "\033[0m"
    DIM = "\033[2m"
    YELLOW = "\033[33m"
    GREEN = "\033[32m"
    CYAN = "\033[36m"
