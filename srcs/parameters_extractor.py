from llm_sdk.llm_sdk import Small_LLM_Model
from typing import Generator, TypeAlias, Callable
from srcs.models import FunctionDefinitions, ParameterType
from srcs.vocab import Vocab

MaskGen: TypeAlias = Generator[tuple[int, ...], int, tuple[int, ...] | None]
MaskFactory: TypeAlias = Callable[[tuple[int, ...]], MaskGen]


class ParameterExtractor:
    def __init__(self, llm: Small_LLM_Model, vocab: Vocab) -> None:
        self.llm = llm
        self.vocab = vocab

        self.MASK: dict[ParameterType, MaskFactory] = {
            "boolean": self._mask_bool
        }

    def _inject(self, prompt: str, chunk: str) -> tuple[str, list[int]]:
        full = self.llm.encode(prompt + chunk)[0].tolist()
        return prompt + chunk, full

    @staticmethod
    def build_prompt(func: FunctionDefinitions, request: str) -> str:
        return (
            "<|im_start|>system"
            "Extract the arguments as JSON.<|im_end|>"
            "<|im_start|>user\n"
            f"Function: {func._function_repr}\n"
            f"Purpose: {func.description}\n"
            f"Request: {request}<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n"
        )

    def _mask_bool(self, stop: tuple[int, ...]) -> MaskGen:
        token = yield self.vocab.true_ids[0], self.vocab.false_ids[0]
        return (
            self.vocab.true_ids[:1]
            if token == self.vocab.true_ids[0]
            else self.vocab.false_ids[:1]
        )

    def _mask_numeric(self, stop: tuple[int, ...], is_integer: bool = False) -> MaskGen:
        dot = False
        sign = False

        while True:
            nb_ids = self.vocab.number_ids
            if not dot and not sign:
                token = yield self.vocab.number_ids + self.vocab.sign_ids
            if sign and not dot:
                token = yield self.vocab.number_ids
            if sign and token in self.vocab.number_ids:
                token = yield self.vocab.number_ids + (self.vocab.dot_id, )
            if dot and token not in self.vocab.number_ids:
                token = yield self.vocab.number_ids
            if dot and token in self.vocab.number_ids:
                token = yield self.vocab.number_ids + stop




    def generate(
        self, context: list[int], type: ParameterType, stop: tuple[int, ...]
    ):
        """
        Take a context as input, its the encoded prompt.
        parameter: type function type
        """

        gen: MaskGen = self.MASK[type](stop)
        generated: list[int] = []
        allowed = next(gen)
        while True:
            logits = self.llm.get_logits_from_input_ids(context + generated)
            token = max(allowed, key=logits.__getitem__)
            if token in stop:
                return generated
            try:
                allowed = gen.send(token)
            except StopIteration as end:
                return generated + list(end.value or ())

    def extract(self, func: FunctionDefinitions, request: str):
        prompt = self.build_prompt(func, request) + "{"
        ids = self.llm.encode(prompt)[0].tolist()
        result: dict[str, str] = {}

        for i, (name, parameter) in enumerate(func.parameters.items()):
            chunk = ("" if i == 0 else " ,") + f'"{name}": '
            if parameter.type == "string":
                chunk += '"'

            prompt, ids = self._inject(prompt, chunk)

            value_ids = self.generate(ids, parameter.type, ())
            result[name] = self.llm.decode(value_ids)
            prompt, ids = self._inject(prompt, result[name])

            if parameter.type == "string":
                prompt += '"'

        return result
