from typing import Literal
from pydantic import BaseModel, RootModel

ParameterType = Literal["number", "string", "boolean", "integer"]


class ParameterSpec(BaseModel):
    type: ParameterType


class FonctionDefinitions(BaseModel):
    name: str
    description: str
    parameters: dict[str, ParameterSpec]


class FonctionCatalog(RootModel[list[FonctionDefinitions]]):
    pass
