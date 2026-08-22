from src.function_selector import FunctionsSelector
from src.parameters_extractor import ParameterExtractor
from src.models import PromptsDefinition, FunctionCatalog

import sys
from typing import Any


class FunctionExtractor:
    """Runs the full pipeline over every prompt of the input file."""

    def __init__(
        self,
        func_selector: FunctionsSelector,
        params_extractor: ParameterExtractor,
        catalog: FunctionCatalog,
        prompts: PromptsDefinition,
    ) -> None:
        """Store the two stages and the data they work on.

        Args:
            func_selector: Picks the function.
            params_extractor: Fills in its arguments.
            catalog: The available functions.
            prompts: The requests to process.
        """
        self.func_selector = func_selector
        self.params_extractor = params_extractor
        self.func_catalog = catalog
        self.prompts = prompts

    def extract(self) -> list[dict[str, Any]]:
        """Turn every prompt into a function call.

        A prompt that fails still gets an entry, so the output stays
        aligned with the input file.

        Returns:
            One object per prompt, in the input order.
        """
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
