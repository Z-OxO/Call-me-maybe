# import numpy as np
from dataclasses import dataclass

# from pydantic.dataclasses import dataclass
from llm_sdk.llm_sdk import Small_LLM_Model
from src.constants import FORBIDDEN_STR


@dataclass(frozen=True, slots=True)
class Vocab:
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
