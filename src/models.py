from typing import Literal
from typing_extensions import Self, Annotated
from pydantic import BaseModel, RootModel, model_validator, Field
from pydantic_core import PydanticCustomError
from collections import Counter
from functools import cached_property

ParameterType = Literal["number", "string", "boolean", "integer"]
NoEmptyStr = Annotated[str, Field(min_length=1)]


class Prompt(BaseModel):
    """One natural-language request read from the input file."""
    prompt: NoEmptyStr


class PromptsDefinition(RootModel[list[Prompt]]):
    """The whole list of requests to process."""
    @model_validator(mode="after")
    def check_empty(self) -> Self:
        """Reject an empty prompt list.

        Returns:
            The validated model.

        Raises:
            ValueError: if the file holds no prompt.
        """
        if not self.root:
            raise ValueError("Prompts test suite cannot be empty")
        return self


class ParameterSpec(BaseModel):
    """The declared type of one function parameter or return value."""
    type: ParameterType


class FunctionDefinitions(BaseModel):
    """One callable function: its name, doc and typed parameters."""
    name: NoEmptyStr
    description: NoEmptyStr
    parameters: dict[NoEmptyStr, ParameterSpec]
    returns: ParameterSpec

    @property
    def _function_repr_desc(self) -> str:
        """One-line signature with the description, for the catalog."""
        args = ", ".join(f"{k}: {v.type}" for k, v in self.parameters.items())
        return f"{self.name}({args}) - {self.description}"

    @property
    def _function_repr(self) -> str:
        """Multi-line signature, used when extracting arguments."""
        args = "\n".join(
            f"  {k}: {v.type}" for k, v in self.parameters.items()
        )
        return f"{self.name} — {self.description}\nParameters:\n{args}"


class FunctionCatalog(RootModel[list[FunctionDefinitions]]):
    """All the functions the model can pick from."""
    @model_validator(mode="after")
    def check_empty(self) -> Self:
        """Reject an empty catalog.

        Returns:
            The validated model.

        Raises:
            ValueError: if the file holds no function.
        """
        if not self.root:
            raise ValueError("functions catalog cannot be empty")
        return self

    @model_validator(mode="after")
    def check_duplication(self) -> Self:
        """Reject duplicate function names.

        Two functions with the same name would make the selection
        ambiguous, and would break the prefix-free property the
        selector relies on.

        Returns:
            The validated model.

        Raises:
            PydanticCustomError: if a name appears more than once.
        """
        names = [function.name for function in self.root]
        dups = [name for name, count in Counter(names).items() if count > 1]
        if dups:
            raise PydanticCustomError(
                "duplicate_names",
                "duplicate function names: {names}",
                {"names": dups},
            )
        return self

    @cached_property
    def catalog_prompt(self) -> str:
        """Every signature, one per line, ready to drop in a prompt."""
        return "\n".join([func._function_repr_desc for func in self.root])

    @cached_property
    def function_choice(self) -> list[str]:
        return [f"{func.name}<|im_end|>" for func in self.root]
