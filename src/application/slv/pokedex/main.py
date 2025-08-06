from .pokemon_encoder import PokemonEncoder
from src.infra.bzr.pokemon.local_pokemon_repository import LocalPokemonRepository
from src.infra.slv.pokedex.local_pokedex_repository import LocalPokedexRepository


def run():
    pokemon_list = LocalPokemonRepository().load_all()

    pokedex_list = []
    for pokemon_item in pokemon_list:
        pokedex_list += PokemonEncoder().run(pokemon_item)

    LocalPokedexRepository().save_all(pokedex_list)


if __name__=="__main__":
    run()