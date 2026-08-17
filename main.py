import sys
from llm_sdk.llm_sdk import Small_LLM_Model
from srcs.models import FunctionCatalog
from pydantic import ValidationError
from srcs.pydantic_errors_format import print_formatted_errors
from srcs.function_selector import FunctionsSelector
from srcs.parameters_extractor import ParameterExtractor
from srcs.vocab import Vocab

if __name__ == "__main__":
    llm: Small_LLM_Model = Small_LLM_Model()
    # print(llm.get_path_to_vocab_file())
    # print(Vocab.from_llm(llm))
    with open("data/input/functions_definition.json") as f:
        try:
            catalog = FunctionCatalog.model_validate_json(f.read())
        except (OSError, ValidationError) as e:
            if isinstance(e, OSError):
                print(f"{e}", file=sys.stderr)
            else:
                print_formatted_errors(e.errors(include_url=False))
            sys.exit()
    selector = FunctionsSelector(llm, catalog)
    extractor = ParameterExtractor(llm, Vocab.from_llm(llm))

    function = selector.choose_fonction("Turn on push notifications")
    print(extractor.extract(function, "Turn on push notifications"))
