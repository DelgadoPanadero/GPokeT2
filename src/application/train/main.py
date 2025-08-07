from datetime import datetime
from src.domain.gld.box_entity import BoxEntity
from src.infra.slv.pokedex import LocalPokedexRepository
from src.infra.gld.prof_oak_pc import LocalProfOakPcRepository

from src.application.gld.prof_oak_pc.tokenizer import Pokenizer
from src.application.train.pokemon_trainer import PokemonTrainer

def run():

    pokedex_list = LocalPokedexRepository().load_all()

    pokenizer = Pokenizer().train(pokedex_list)
    dataset = pokenizer.tokenize(pokedex_list)

    box_entity = BoxEntity(
        name = datetime.now().strftime("%Y%m%d-%H%M"),
        tokenizer=pokenizer.to_dict(),
        dataset=dataset,
    )

    PokemonTrainer(box_entity).create_trainer().train()
    

if __name__=="__main__":
    run()