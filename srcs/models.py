from typing import Literal
from typing_extensions import Self, Annotated
from pydantic import BaseModel, RootModel, model_validator, Field
from collections import Counter
from srcs.constants import Colors

ParameterType = Literal["number", "string", "boolean", "integer"]
NoEmptyStr = Annotated[str, Field(min_length=1)]


class ParameterSpec(BaseModel):
    type: ParameterType


class FonctionDefinitions(BaseModel):
    name: NoEmptyStr
    description: NoEmptyStr
    parameters: dict[NoEmptyStr, ParameterSpec]
    returns: ParameterSpec


class FonctionCatalog(RootModel[list[FonctionDefinitions]]):
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
            raise ValueError(
                f"Duplicate function names: "
                f"{Colors.CYAN}{', '.join(dups)}{Colors.RESET}"
            )
        return self
