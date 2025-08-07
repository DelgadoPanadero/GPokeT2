from pydantic import BaseModel
from datasets import DatasetDict


class BoxEntity(BaseModel):
    name: str
    tokenizer: dict
    dataset: DatasetDict

    model_config = {"arbitrary_types_allowed": True}
