import cv2
import numpy as np
import numpy.typing as npt
from pathlib import Path
from ..domain.pokemon_entity import PokemonEntity


class LocalPokemonRepository():

    def load_one(
        self,
        img_path: Path,
    ) -> npt.NDArray[np.int_]:

        image = cv2.imread(str(img_path))
        image = np.array(image)
        return image


    def load_all(
        self,
    )-> list[npt.NDArray[np.int_]]:

        source_dir = "/home/data/bzr/pokemons_img/"
        paths = [x for x in Path(source_dir).glob("**/*.png")]

        result = []
        for path in paths:
            result.append(self.load_one(path))

        return result


