from src.function_selector import FunctionsSelector
from src.parameters_extractor import ParameterExtractor
from src.models import PromptsDefinition, FunctionCatalog

import sys
from typing import Any


class FunctionExtractor:
    def __init__(
        self,
        func_selector: FunctionsSelector,
        params_extractor: ParameterExtractor,
        catalog: FunctionCatalog,
        prompts: PromptsDefinition,
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
                results.append(
                    {
                        "prompt": case.prompt,
                        "name": func.name,
                        "parameters": args,
                    }
                )
            except Exception as e:
                print(f"{case.prompt!r}: {e}", file=sys.stderr)
                results.append({"prompt": case.prompt, "error": str(e)})
        return results
