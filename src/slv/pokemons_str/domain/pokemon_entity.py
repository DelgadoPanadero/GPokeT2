import numpy as np
import numpy.typing as npt
from pydantict import BaseModel


class PokemonEntity(BaseModel):
    name: str
    image: npt.NDArray[np.int8]