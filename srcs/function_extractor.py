from srcs.function_selector import FunctionsSelector, FunctionCatalog
from srcs.parameters_extractor import ParameterExtractor
from srcs.models import PromptsDefinition, FunctionCatalog

import sys
from typing import Any
from time import time


class FunctionExtractor:
    def __init__(
        self,
        func_selector: FunctionsSelector,
        params_extractor: ParameterExtractor,
        catalog: FunctionCatalog,
        prompts: PromptsDefinition
    ) -> None:
        self.func_selector = func_selector
        self.params_extractor = params_extractor
        self.func_catalog = catalog
        self.prompts = prompts

    def extract(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for case in self.prompts.root:
            try:
                func = self.func_selector.choose_fonction(case.prompt)
                args = self.params_extractor.extract(func, case.prompt)
                results.append({"name": func.name, "arguments": args})
            except Exception as e:
                print(f"{case.prompt!r}: {e}", file=sys.stderr)
                results.append({"error": f"{type(e).__name__}: {e}"})
        return results
