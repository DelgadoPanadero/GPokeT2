from src.infra.gld.prof_oak_pc import LocalProfOakPcRepository
from src.application.train.pokemon_trainer import PokemonTrainer


def run():

    box_entity = LocalProfOakPcRepository().load(box_name="box-20250806-1843")

    PokemonTrainer(box_entity).create_trainer().train()
    

if __name__=="__main__":
    run()