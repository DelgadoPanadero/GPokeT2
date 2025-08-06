from datetime import datetime
from src.domain.gld.box_entity import BoxEntity
from src.infra.slv.pokedex import LocalPokedexRepository
from src.infra.gld.prof_oak_pc import LocalProfOakPcRepository

from src.application.gld.prof_oak_pc.tokenizer import Pokenizer


def run():

    pokedex_list = LocalPokedexRepository().load_all()

    pokenizer = Pokenizer().train(pokedex_list)
    dataset = pokenizer.tokenize(pokedex_list)

    box_entity = BoxEntity(
        name = datetime.now().strftime("%Y%m%d-%H:%M"),
        tokenizer=pokenizer.to_dict(),
        dataset=dataset,
    )

    LocalProfOakPcRepository().save(box_entity)
    

if __name__=="__main__":
    run()