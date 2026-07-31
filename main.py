import sys
from llm_sdk.llm_sdk import Small_LLM_Model
from srcs.models import FunctionCatalog
from pydantic import ValidationError
from srcs.pydantic_errors_format import print_formatted_errors
from srcs.function_selector import FunctionsSelector

if __name__ == "__main__":
    llm: Small_LLM_Model = Small_LLM_Model()
    with open("data/input/functions_definition.json") as f:
        try:
            catalog = FunctionCatalog.model_validate_json(f.read())
        except (OSError, ValidationError) as e:
            if isinstance(e, OSError):
                print(f"{e}", file=sys.stderr)
            else:
                print_formatted_errors(e.errors(include_url=False))
            exit()
    selector = FunctionsSelector(llm, catalog)
    print(
        selector.choose_fonction(
            "Whats the sum of 2 , 2 , 2 , 2",
        )
    )
