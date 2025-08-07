from datetime import datetime

from src.domain.gld.prof_oak_pc import BoxEntity
from src.domain.slv.pokedex import PokedexRepository
from src.domain.gld.prof_oak_pc import ProfOakPcRepository
from src.application.gld.prof_oak_pc.tokenizer import Pokenizer



def get_prod_oak_pc(
    pokedex_repository: PokedexRepository,
    profoakpc_repository: ProfOakPcRepository,
)->None:

    pokedex_list = pokedex_repository.load_all()

    pokenizer = Pokenizer().train(pokedex_list)
    dataset = pokenizer.tokenize(pokedex_list)

    box_entity = BoxEntity(
        name = "box-" + datetime.now().strftime("%Y%m%d-%H:%M"),
        tokenizer=pokenizer.to_dict(),
        dataset=dataset,
    )

    profoakpc_repository.save(box_entity)


if __name__=="__main__":
    
    from src.infra.slv.pokedex import LocalPokedexRepository
    from src.infra.gld.prof_oak_pc import LocalProfOakPcRepository

    get_prod_oak_pc(
        pokedex_repository=LocalPokedexRepository(),
        profoakpc_repository=LocalProfOakPcRepository(),
    )