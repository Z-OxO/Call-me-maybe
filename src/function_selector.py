from src.models import FunctionCatalog, FunctionDefinitions
from llm_sdk.llm_sdk import Small_LLM_Model


class FunctionsSelector:
    """Picks the function that answers a request.

    The name is generated token by token under a trie constraint, so
    the model can only ever spell out a name that exists.
    """

    def __init__(self, llm: Small_LLM_Model, catalog: FunctionCatalog) -> None:
        """Store the model and pre-encode every function name.

        Each name gets "<|im_end|>" appended before encoding. That
        makes the set prefix-free, so a name that is a prefix of
        another one stays reachable.

        Args:
            llm: The model used to encode and score tokens.
            catalog: The functions to choose from.
        """
        self.llm = llm
        self.catalog = catalog
        self.encoded = [
            llm.encode(f"{function.name}<|im_end|>")[0].tolist()
            for function in catalog.root
        ]

    def _build_prompt(self, request: str) -> str:
        """Build the chat prompt listing the functions and the request."""
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
        """Generate a function name under a trie constraint.

        At each step we group the still-matching names by their next
        token. If they all agree, that token is forced and we skip the
        forward pass. If they disagree, we run the model and keep the
        highest-scoring branch. Either way the result is always a real
        function name.

        Args:
            request: The natural-language request.

        Returns:
            The chosen function definition.
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
