from typing import Literal
from typing_extensions import Self, Annotated
from pydantic import BaseModel, RootModel, model_validator, Field
from pydantic_core import PydanticCustomError
from collections import Counter
from functools import cached_property

ParameterType = Literal["number", "string", "boolean", "integer"]
NoEmptyStr = Annotated[str, Field(min_length=1)]


class ParameterSpec(BaseModel):
    type: ParameterType


class FunctionDefinitions(BaseModel):
    name: NoEmptyStr
    description: NoEmptyStr
    parameters: dict[NoEmptyStr, ParameterSpec]
    returns: ParameterSpec

    @property
    def _function_repr_desc(self) -> str:
        args = ", ".join(f"{k}: {v.type}" for k, v in self.parameters.items())
        return f"{self.name}({args}) - {self.description}"

    @property
    def _function_repr(self) -> str:
        args = ", ".join(f"{k}: {v.type}" for k, v in self.parameters.items())
        return f"{self.name}({args}) - {self.description}"


class FunctionCatalog(RootModel[list[FunctionDefinitions]]):
    @model_validator(mode="after")
    def check_empty(self) -> Self:
        if not self.root:
            raise ValueError("functions catalog cannot be empty")
        return self

    @model_validator(mode="after")
    def check_duplication(self) -> Self:
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
        return "\n".join([func._function_repr_desc for func in self.root])

    @cached_property
    def function_choice(self) -> list[str]:
        return [f"{func.name}<|im_end|>" for func in self.root]
