from dataclasses import dataclass


@dataclass
class Colors:
    RESET = "\033[0m"
    DIM = "\033[2m"
    YELLOW = "\033[33m"
    GREEN = "\033[32m"
    CYAN = "\033[36m"
