from .tokenizer import Pokenizer
from .dataset import PokeCenter
from src.infra.slv.pokedex import LocalPokedexRepository


def run():
    pokedex_list = LocalPokedexRepository().load_all()

    pokenizer = Pokenizer().train(pokedex_list)
    pokecenter = PokeCenter

if __name__=="__main__":
    run()