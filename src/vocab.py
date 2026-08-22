# import numpy as np
from dataclasses import dataclass

# from pydantic.dataclasses import dataclass
from llm_sdk.llm_sdk import Small_LLM_Model
from src.constants import FORBIDDEN_STR


@dataclass(frozen=True, slots=True)
class Vocab:
    """Token ids the JSON masks need, resolved once for a tokenizer.

    Built from the model, then kept immutable: the object no longer
    depends on the SDK once created.
    """

    number_ids: tuple[int, ...]
    str_forbidden: tuple[int, ...]
    sign_ids: tuple[int, ...]
    true_ids: tuple[int, ...]
    false_ids: tuple[int, ...]
    dot_id: int
    quote_id: int
    comma_id: int
    brace_close_id: int

    @classmethod
    def from_llm(cls, llm: Small_LLM_Model) -> "Vocab":
        """Look up every token id the masks rely on.

        Args:
            llm: The model whose tokenizer is used.

        Returns:
            A Vocab holding the digit, sign, literal and punctuation
            token ids.

        Raises:
            ValueError: if the tokenizer splits one of the required
                characters into several tokens. The masks work on
                single ids, so they cannot express that case.
        """

        numbers: tuple[int, ...] = (*llm.encode("0123456789")[0].tolist(),)

        str_forbidden = tuple(
            (llm.encode(c)[0].tolist()[0] for c in FORBIDDEN_STR)
        )
        signs = llm.encode("-")[0].tolist()[0], llm.encode("+")[0].tolist()[0]
        true: tuple[int, ...] = tuple(llm.encode("true")[0].tolist())
        false: tuple[int, ...] = tuple(llm.encode("false")[0].tolist())
        dot: int = llm.encode(".")[0].tolist()[0]
        quote: int = llm.encode('"')[0].tolist()[0]
        comma: int = llm.encode(",")[0].tolist()[0]
        brace_close: int = llm.encode("}")[0].tolist()[0]

        return cls(
            numbers,
            str_forbidden,
            signs,
            true,
            false,
            dot,
            quote,
            comma,
            brace_close,
        )
