import os
from ..domain.pokedex_entity import PokedexEntity


class LocalPokedexRepository():
    def save_one(
        self,
        pokedex_item: PokedexEntity,
    ):
        source_dir = "/home/data/slv/pokedex"
        os.makedirs(source_dir,exist_ok=True)
        with open(f"{source_dir}/{pokedex_item.name}", "w") as file:
            file.write(pokedex_item.data)
        
    def save_all(
        self,
        pokedex_list: list[PokedexEntity]
    ):

        for pokedex_item in pokedex_list:
            self.save_one(pokedex_item)