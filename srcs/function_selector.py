from srcs.models import FunctionCatalog, FunctionDefinitions
from llm_sdk.llm_sdk import Small_LLM_Model


class FunctionsSelector:
    def __init__(self, llm: Small_LLM_Model, catalog: FunctionCatalog) -> None:
        self.llm = llm
        self.catalog = catalog
        self.encoded = [
            llm.encode(f"{function.name}<|im_end|>")[0].tolist()
            for function in catalog.root
        ]

    def _build_prompt(self, request: str) -> str:
        return (
            "<|im_start|>system\n"
            "You select the function that answers the user request. "
            "Reply with the function name only.<|im_end|>\n"
            "<|im_start|>user\n"
            f"Available functions:\n{self.catalog.catalog_prompt}\n\n"
            f"Request: {request}\n"
            "Which function?<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n"
        )

    def choose_fonction(self, request: str) -> FunctionDefinitions:
        """
        Use a tree to represent the tokens options (branches),
        minimal forward by construction.
        If there is one choice of token we skip the forward
        Acces to the first element of the dico by
        iter -> transform the dict to a generator (by the key)
        -> next() -> take the unique element of branches)
        """
        context = self.llm.encode(self._build_prompt(request))[0].tolist()

        generated: list[int] = []
        alive = list(range(len(self.encoded)))

        while len(alive) > 1:
            branches: dict[int, list[int]] = {}
            for i in alive:
                branches.setdefault(
                    self.encoded[i][len(generated)], []
                ).append(i)
            if len(branches) == 1:
                token = next(iter(branches))
            else:
                logits = self.llm.get_logits_from_input_ids(
                    context + generated
                )
                token = max(branches, key=logits.__getitem__)

            generated.append(token)
            alive = branches[token]

        return self.catalog.root[alive[0]]
