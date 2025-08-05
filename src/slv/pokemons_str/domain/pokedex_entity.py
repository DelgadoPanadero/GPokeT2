from pydantict import BaseModel


class PokedexEntity(BaseModel):
    name: str
    image: list[list[int]]