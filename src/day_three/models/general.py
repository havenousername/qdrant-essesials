from enum import Enum, StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_serializer


class ProhibitionTiming(StrEnum):
    ALL_TIMES = "all_times"
    IN_COMPETITION = "in_competition"
    PARTICULAR_SPORTS = "in_particular_sports"


class WadaCategory(Enum):
    # Substances — prohibited at all times
    S0 = ("S0", "Non-Approved Substances", ProhibitionTiming.ALL_TIMES)
    S1 = ("S1", "Anabolic Agents", ProhibitionTiming.ALL_TIMES)
    S2 = ("S2", "Peptide Hormones, Growth Factors, Related Substances and Mimetics", ProhibitionTiming.ALL_TIMES)
    S3 = ("S3", "Beta-2 Agonists", ProhibitionTiming.ALL_TIMES)
    S4 = ("S4", "Hormone and Metabolic Modulators", ProhibitionTiming.ALL_TIMES)
    S5 = ("S5", "Diuretics and Masking Agents", ProhibitionTiming.ALL_TIMES)
    # Methods — prohibited at all times
    M1 = ("M1", "Manipulation of Blood and Blood Components", ProhibitionTiming.ALL_TIMES)
    M2 = ("M2", "Chemical and Physical Manipulation", ProhibitionTiming.ALL_TIMES)
    M3 = ("M3", "Gene and Cell Doping", ProhibitionTiming.ALL_TIMES)
    # Substances — prohibited in competition only
    S6 = ("S6", "Stimulants", ProhibitionTiming.IN_COMPETITION)
    S7 = ("S7", "Narcotics", ProhibitionTiming.IN_COMPETITION)
    S8 = ("S8", "Cannabinoids", ProhibitionTiming.IN_COMPETITION)
    S9 = ("S9", "Glucocorticoids", ProhibitionTiming.IN_COMPETITION)
    # Substances — prohibited in particular sports only
    P1 = ("P1", "Beta-Blockers", ProhibitionTiming.PARTICULAR_SPORTS)

    def __init__(self, code: str, label: str, timing: ProhibitionTiming) -> None:
        self.code = code
        self.label = label
        self.timing = timing

    def __str__(self) -> str:
        return f"{self.code} - {self.label}"

    @classmethod
    def _missing_(cls, value):
        if not isinstance(value, str):
            return None

        parts = value.split(" - ", 1)

        if len(parts) != 2:
            return None

        code, _ = parts

        for category in cls:
            if category.code == code:
                return category

        return None


class DataChunk[T](BaseModel):
    chunk: list[T]
    index: int
    position: int = Field(default=0)


class TextSubstance(BaseModel):
    name: str
    drug_category: str
    description: str
    sections: list[str]
    section_names: list[str]

    def to_natural_text(self) -> str:
        sections_text = '\n'.join([
            f'Section {idx}: {section}' for idx, section in enumerate(self.sections)
        ])
        return (f'Subtance name is {self.name}.'
            f'It is in the {self.drug_category} category.'
            f'Short description is: "{self.description}".\n'
            f'{sections_text}'
        ) 


WadaStatus = Literal["banned", "conditional", "legal"]

class RawSubstance(BaseModel):
    name: str
    description: str
    drug_category: str
    wada_status: WadaStatus
    wada_category: WadaCategory | None = Field(default=None)
    side_effects: str | None = Field(default=None)

    @field_serializer('wada_category')
    def serialize_wada_category(self, wada_category: WadaCategory | None):
        return str(wada_category) if wada_category else None

    def to_natural_text(self) -> str:
        return (
            f'Subtance name is {self.name} from category {self.drug_category}'
            f'{self.description}\n'
            f'Wada information: status is {self.wada_status} and {self.wada_category}\n\n'
            f'Side effects are {self.side_effects}'
        )