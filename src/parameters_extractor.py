import sys
from typing import Generator, TypeAlias, Callable, Any
from functools import partial

from llm_sdk.llm_sdk import Small_LLM_Model
from src.models import FunctionDefinitions, ParameterType
from src.vocab import Vocab
from src.constants import FORBIDDEN_STR, MAX_TOKEN

MaskGen: TypeAlias = Generator[
    tuple[int, ...] | None, int, tuple[int, ...] | None
]
MaskFactory: TypeAlias = Callable[[tuple[int, ...]], MaskGen]


class ParameterExtractor:
    """Fills in the arguments of a function, one value at a time.

    The JSON skeleton (braces, keys, quotes, commas) is injected
    directly into the prompt and costs no forward pass. Only the
    values are generated, each under a mask that fits its type.
    """

    def __init__(self, llm: Small_LLM_Model, vocab: Vocab) -> None:
        """Store the model and vocab, and map each type to its mask.

        Args:
            llm: The model used to encode and score tokens.
            vocab: The token ids the masks need.
        """
        self.llm = llm
        self.vocab = vocab

        self.MASK: dict[ParameterType, MaskFactory] = {
            "boolean": self._mask_bool,
            "number": self._mask_numeric,
            "integer": partial(self._mask_numeric, is_integer=True),
            "string": self._mask_strings,
        }

    def _inject(self, prompt: str, chunk: str) -> tuple[str, list[int]]:
        """Append a chunk to the prompt and re-encode the whole thing."""
        full = self.llm.encode(prompt + chunk)[0].tolist()
        return prompt + chunk, full

    @staticmethod
    def build_prompt(func: FunctionDefinitions, request: str) -> str:
        """Build the chat prompt for one function and one request.

        Args:
            func: The function whose arguments we want.
            request: The natural-language request.

        Returns:
            The prompt, ready for the opening brace.
        """
        return (
            "<|im_start|>system\n"
            "Extract the function arguments from the request as JSON.\n"
            "Copy values from the request. "
            "Use literal characters, not their names.\n"
            "Keep values short: no regex groups, no alternation.\n"
            "Example:\n"
            "Request: Replace all letters in 'a1b2' with LETTERS\n"
            '{"source_string": "a1b2", "regex": "[a-z]", '
            '"replacement": "LETTERS"}\n'
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"{func._function_repr}\n"
            f"Request: {request}<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n"
        )

    def _pick(self, logits: list[float], ends: dict[str, int]) -> int:
        """Pick the best token that keeps the JSON string valid.

        Tries the highest-scoring token, and rejects it if it decodes
        to nothing (a special token) or holds a forbidden character.
        A token whose first character closes the value is turned into
        the matching stop token, which handles the ones BPE merged
        with something else.

        Args:
            logits: The scores for this step. Modified in place.
            ends: First character to stop token.

        Returns:
            The id of the accepted token, or of a stop token.
        """
        while True:
            token = max(range(len(logits)), key=logits.__getitem__)
            piece = self.llm.decode([token])
            if not piece:
                logits[token] = float("-inf")
                continue
            stop = ends.get(piece[:1])
            if stop is not None:
                return stop
            if FORBIDDEN_STR.isdisjoint(piece):
                return token
            logits[token] = float("-inf")

    def _mask_strings(self, stop: tuple[int, ...]) -> MaskGen:
        """Allow anything: _pick filters the tokens for strings."""
        while True:
            yield None

    def _mask_bool(self, stop: tuple[int, ...]) -> MaskGen:
        """Offer "true" and "false", then inject the rest of the word."""
        token = yield self.vocab.true_ids[0], self.vocab.false_ids[0]
        return (
            self.vocab.true_ids[1:]
            if token == self.vocab.true_ids[0]
            else self.vocab.false_ids[1:]
        )

    def _mask_numeric(
        self, stop: tuple[int, ...], is_integer: bool = False
    ) -> MaskGen:
        """Yield the tokens allowed at each step of a number.

        Signs come first only, a dot only after a digit and only once,
        and stopping is offered only when what we have is already a
        valid number. Integers never get the dot.

        Args:
            stop: Tokens that end the value.
            is_integer: Whether to forbid the decimal point.
        """
        started = dot = digit = digit_after_dot = False
        while True:
            allowed = self.vocab.number_ids
            if not started:
                allowed += self.vocab.sign_ids
            if digit and not dot and not is_integer:
                allowed += (self.vocab.dot_id,)
            if digit and (not dot or digit_after_dot):
                allowed += stop
            token = yield allowed
            started = True
            if token == self.vocab.dot_id:
                dot = True
            elif token in self.vocab.number_ids:
                digit_after_dot = dot
                digit = True

    @staticmethod
    def _format_result(
        func: FunctionDefinitions, raw: dict[str, str]
    ) -> dict[str, Any]:
        """Convert the generated text to the types the schema asks for.

        Args:
            func: The function the values belong to.
            raw: Generated text, per parameter name.

        Returns:
            The parameters with their real Python types.
        """
        out: dict[str, Any] = {}
        for name, spec in func.parameters.items():
            text = raw[name]
            match spec.type:
                case "number":
                    out[name] = float(text)
                case "integer":
                    out[name] = int(text)
                case "boolean":
                    out[name] = text == "true"
                case _:
                    out[name] = text
        return out

    def generate(
        self, context: list[int], type: ParameterType, stop: tuple[int, ...]
    ) -> list[int]:
        """Generate one value, token by token, under its type mask.

        Stops on a stop token, or when the token budget runs out. In
        that second case the value is cut back to the last point where
        the mask allowed stopping, so what comes out always parses as
        its type. Without that we could return "12." or a lone "-".

        Args:
            context: The encoded prompt to generate from.
            type: The parameter type, which selects the mask.
            stop: Tokens that end the value.

        Returns:
            The generated token ids.
        """
        gen: MaskGen = self.MASK[type](stop)
        generated: list[int] = []
        ends = {self.llm.decode([t]): t for t in stop}
        allowed = next(gen)
        safe = 0

        while True:
            if allowed is None or not set(stop).isdisjoint(allowed):
                safe = len(generated)
            if len(generated) >= MAX_TOKEN:
                return generated[:safe]

            logits = self.llm.get_logits_from_input_ids(context + generated)
            if allowed is None:
                token = self._pick(logits, ends)
            else:
                token = max(allowed, key=logits.__getitem__)
            if token in stop:
                return generated
            generated.append(token)
            try:
                allowed = gen.send(token)
            except StopIteration as end:
                return generated + list(end.value or ())

    def extract(
        self, func: FunctionDefinitions, request: str
    ) -> dict[str, Any]:
        """Fill in every argument of one function call.

        Walks the parameters in order, injecting the JSON around each
        value and generating only the value itself. The prompt grows
        as we go, so each value is chosen knowing the previous ones.

        Args:
            func: The function to fill in.
            request: The natural-language request.

        Returns:
            The parameters, typed as the schema declares them.
        """
        prompt = self.build_prompt(func, request) + "{"
        ids = self.llm.encode(prompt)[0].tolist()
        result: dict[str, str] = {}

        for i, (name, parameter) in enumerate(func.parameters.items()):
            chunk = ("" if i == 0 else " ,") + f'"{name}": '
            if parameter.type == "string":
                chunk += '"'
                stop = (self.vocab.quote_id,)
            else:
                stop = (self.vocab.comma_id, self.vocab.brace_close_id)

            prompt, ids = self._inject(prompt, chunk)

            try:
                value_ids = self.generate(ids, parameter.type, stop)
            except ValueError as e:
                print(f"{func.name}.{name}: {e}", file=sys.stderr)
                value_ids = []

            result[name] = self.llm.decode(value_ids)
            prompt, ids = self._inject(prompt, result[name])
            if parameter.type == "string":
                prompt += '"'

        return self._format_result(func, result)
