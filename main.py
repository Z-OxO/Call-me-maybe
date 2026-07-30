from llm_sdk.llm_sdk import Small_LLM_Model
from srcs.models import FonctionCatalog
from pydantic import ValidationError
from srcs.pydantic_errors_format import print_formatted_errors

CATALOG = """fn_add_numbers(a: number, b: number) - Add two numbers together and return their sum.
fn_add_numbers_list(list: list[number]) - Add a list of numbers together and return their sum.
fn_greet(name: string) - Generate a greeting message for a person by name.
fn_reverse_string(s: string) - Reverse a string and return the reversed result.
fn_get_square_root(a: number) - Calculate the square root of a number.
fn_substitute_string_with_regex(source_string: string, regex: string, replacement: string) - Replace all occurrences matching a regex pattern in a string."""

NAMES = [
    "fn_add_numbers<|im_end|>",
    "fn_greet<|im_end|>",
    "fn_reverse_string<|im_end|>",
    "fn_get_square_root<|im_end|>",
    "fn_substitute_string_with_regex<|im_end|>",
    "fn_add_numbers_list<|im_end|>",
]


def build_prompt(request: str) -> str:
    return (
        "<|im_start|>system\n"
        "You select the function that answers the user request. "
        "Reply with the function name only.<|im_end|>\n"
        "<|im_start|>user\n"
        f"Available functions:\n{CATALOG}\n\n"
        f"Request: {request}\n"
        "Which function?<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
    )


def choose_fonction(
    question: str, llm: Small_LLM_Model, names: list[str]
) -> tuple[str, int]:
    """
    Use a tree to represent the tokens options (branches), minimal forward by
    construction.
    If there is one choice of token we skip the forward
    Acces to the first element of the dico by
    iter -> transform the dict to a generator (by the key)
    -> next() -> take the unique element of branches)
    """
    context = llm.encode(build_prompt(question))[0].tolist()
    encoded = [llm.encode(name)[0].tolist() for name in names]

    generated: list[int] = []
    alive = list(range(len(encoded)))
    forwards = 0

    while len(alive) > 1:
        branches: dict[int, list[int]] = {}
        for i in alive:
            branches.setdefault(encoded[i][len(generated)], []).append(i)

        if len(branches) == 1:
            token = next(iter(branches))
        else:
            logits = llm.get_logits_from_input_ids(context + generated)
            forwards += 1
            token = max(branches, key=logits.__getitem__)

        generated.append(token)
        alive = branches[token]

    return llm.decode(generated).removesuffix("<|im_end|>"), forwards


if __name__ == "__main__":
    # llm: Small_LLM_Model = Small_LLM_Model()
    with open("data/input/functions_definition.json") as f:
        try:
            catalog = FonctionCatalog.model_validate_json(f.read())
            print(catalog)
        except (OSError, ValidationError) as e:
            if isinstance(e, OSError):
                print(f"{e}")
            else:
                print_formatted_errors(e.errors(include_url=False))
    # print(choose_fonction("Whats the sum of 2 , 2 , 2 , 2", llm, NAMES))
