import json
from typing import Any


class Parser:
    @staticmethod
    def parse_file_json(file_path: str) -> Any | None:
        try:
            with open(file_path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
            print(f"{file_path}: {e}")
            return None
