import sys
import json
import argparse
from llm_sdk.llm_sdk import Small_LLM_Model
from srcs.models import FunctionCatalog, PromptsDefinition
from pydantic import ValidationError
from time import time
from pathlib import Path


from srcs.pydantic_errors_format import print_formatted_errors
from srcs.function_selector import FunctionsSelector
from srcs.parameters_extractor import ParameterExtractor
from srcs.function_extractor import FunctionExtractor
from srcs.vocab import Vocab


def load_catalog(path: Path) -> FunctionCatalog:
    with open(path) as f:
        return FunctionCatalog.model_validate_json(f.read())


def load_prompts(path: Path) -> PromptsDefinition:
    with open(path) as f:
        return PromptsDefinition.model_validate_json(f.read())


def main() -> int:

    parser = argparse.ArgumentParser()
    parser.add_argument("--functions_definition")
    parser.add_argument("--input")
    parser.add_argument("--output")

    args = parser.parse_args()
    try:
        catalog = load_catalog(Path(args.functions_definition))
        prompts = load_prompts(Path(args.input))
    except OSError as e:
        print(e, file=sys.stderr)
        return 1
    except ValidationError as e:
        print_formatted_errors(e.errors(include_url=False))
        return 1
    llm = Small_LLM_Model()
    time_start = time()
    extractor = FunctionExtractor(
        FunctionsSelector(llm, catalog),
        ParameterExtractor(llm, Vocab.from_llm(llm)),
        catalog,
        prompts,
    )
    results = extractor.extract()
    time_end = time()
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Temps: {time_end - time_start}")
    return 0


if __name__ == "__main__":
    p = cProfile.Profile()
    p.runcall(main)
    pstats.Stats(p).sort_stats("tottime").print_stats(30)
