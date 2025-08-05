import numpy as np
import numpy.typing as npt

from ..domain.pokedex_entity import PokedexEntity
from ..domain.pokemon_entity import PokemonEntity


class PokemonEncoder:

    def __init__(
        self,
        n_characters: int = 64,
    ):
        self.n_characters = n_characters


    @staticmethod
    def _encode(
        image: npt.NDArray[np.int8],
    )-> list[list[str]]:
        
        """
        """

        width, height, _ = image.shape

        array = []
        for y in range(0, height):

            row = ["%02d" % 4]
            for x in range(0, width, 2):
                r, g, b = image[y, x] // 64
                is_blank = min(image[y, x]) > 245 or max(image[y, x]) < 10
                char = (
                    "~"
                    if is_blank
                    else chr(r * 4**2 + g * 4**1 + b * 4**0 + 59)
                )

                row.append(char)
            array.append(row[:-1])

        return array

    @staticmethod
    def _array_to_text(
        array : list[list[str]],
    )->str:
        return "\n".join([" ".join(r) for i, r in enumerate(array)])


    @classmethod
    def _augmentation(cls, array):
        """ """

        batch = {}
        batch["original"] = array
        batch["fliped"] = [row[0:1] + row[:0:-1] for row in array]
        # TODO color transformation

        return batch

    def run(
        self,
        pokemon: PokemonEntity,
    )-> list[PokedexEntity]:
        
        image = pokemon.image

        array = self._encode(image)
        batch = self._augmentation(array)

        result = []
        for name, array in batch.items():
            pokedex_name = pokemon.name.replace('.png',f'_{name}.txt')
            pokedex_data = self._array_to_text(array)
            result.append(
                PokedexEntity(
                    name=pokedex_name,
                    data=pokedex_data,
                ),
            )

        return result


