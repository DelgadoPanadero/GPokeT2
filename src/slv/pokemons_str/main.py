import cv2
from pathlib import Path

from .application.pokemon_encoder import PokemonEncoder
from .infra.local_pokemon_img_repository import LocalPokemonImgRepository
from .infra.local_pokemon_img_repository import PokemonImgRepository

def run():
    data = LocalPokemonImgRepository().load_all()

    for item in data:
         PokemonEncoder().encode(item)

    for name, array in outputs.items():
        file_name = (img_name
                    .replace(".png", f"_{name}.txt")
                    .replace("pokemons/", "pokemons_txt/")
                )
                with open(file_name, "w") as file:
                    text = cls.array_to_text(array)
                    file.write(text)