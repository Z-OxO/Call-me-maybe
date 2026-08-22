import sys
import json
import argparse
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from llm_sdk.llm_sdk import Small_LLM_Model
from src.models import FunctionCatalog, PromptsDefinition
from src.pydantic_errors_format import print_formatted_errors
from src.function_selector import FunctionsSelector
from src.parameters_extractor import ParameterExtractor
from src.function_extractor import FunctionExtractor
from src.constants import DEFAULT_FUNCTIONS, DEFAULT_INPUT, DEFAULT_OUTPUT
from src.vocab import Vocab


def load_catalog(path: Path) -> FunctionCatalog:
    """Read and validate the function definitions file.

    Args:
        path: Path to the JSON file.

    Returns:
        The validated catalog.

    Raises:
        OSError: if the file cannot be read.
        ValidationError: if the content does not match the schema.
    """
    with open(path, encoding="utf-8") as f:
        return FunctionCatalog.model_validate_json(f.read())


def load_prompts(path: Path) -> PromptsDefinition:
    """Read and validate the prompts file.

    Args:
        path: Path to the JSON file.

    Returns:
        The validated prompt list.

    Raises:
        OSError: if the file cannot be read.
        ValidationError: if the content does not match the schema.
    """
    with open(path, encoding="utf-8") as f:
        return PromptsDefinition.model_validate_json(f.read())


def write_results(path: Path, results: list[dict[str, Any]]) -> None:
    """Write the function calls, creating the directory if needed.

    Args:
        path: Where to write the JSON file.
        results: One object per prompt.

    Raises:
        OSError: if the file cannot be written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        f.write("\n")


def parse_args() -> argparse.Namespace:
    """Parse the command line, falling back to the data/ layout."""
    parser = argparse.ArgumentParser(prog="src")
    parser.add_argument(
        "--functions_definition", type=Path, default=DEFAULT_FUNCTIONS
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    """Load the inputs, run the pipeline, write the results.

    Returns:
        0 on success, 1 if anything went wrong.
    """
    args = parse_args()

    try:
        catalog = load_catalog(args.functions_definition)
        prompts = load_prompts(args.input)
    except OSError as e:
        print(e, file=sys.stderr)
        return 1
    except UnicodeDecodeError:
        print("input files must be UTF-8 encoded", file=sys.stderr)
        return 1
    except ValidationError as e:
        print_formatted_errors(e.errors(include_url=False))
        return 1

    try:
        llm = Small_LLM_Model()
        vocab = Vocab.from_llm(llm)
    except Exception as e:
        print(f"model setup failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    extractor = FunctionExtractor(
        FunctionsSelector(llm, catalog),
        ParameterExtractor(llm, vocab),
        catalog,
        prompts,
    )
    results = extractor.extract()

    try:
        write_results(args.output, results)
    except OSError as e:
        print(e, file=sys.stderr)
        return 1

    print(f"{len(results)} call(s) written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
