from enum import StrEnum

from datasets import load_dataset


class DatasetNames(StrEnum): 
    WOLT_FOOD = "Qdrant/wolt-food-clip-ViT-B-32-embeddings"

wolt_food_ds = load_dataset(DatasetNames.WOLT_FOOD)