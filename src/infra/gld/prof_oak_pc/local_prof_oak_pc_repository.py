import os
import json
from tokenizers import Tokenizer
from datasets import DatasetDict
from src.domain.gld.box_entity import BoxEntity


class LocalProfOakPcRepository():

    def save(
        self,
        box_entity: BoxEntity
    ):
        source_dir = f"/home/data/gld/prof_oak_pc/{box_entity.name}"

        os.makedirs(source_dir,exist_ok=True)

        # Save dataset
        box_entity.dataset.save_to_disk(source_dir)

        # Save tokenizer
        with open(f"{source_dir}/tokenizer.json", "w") as fin:
            fin.write(
                json.dumps(
                    box_entity.tokenizer,
                    indent=4,
                    ensure_ascii=False,
                )
            )

    def load(
        self,
        box_name: str,
    )->BoxEntity:
        source_dir = f"/home/data/gld/prof_oak_pc/{box_name}"

        dataset = DatasetDict.load_from_disk(source_dir)
        with open(f"{source_dir}/tokenizer.json", "r") as fin:
            tokenizer = json.load(fin)

        return BoxEntity(
            name = box_name,
            dataset=dataset,
            tokenizer=tokenizer,
        )
