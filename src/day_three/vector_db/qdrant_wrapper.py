

import os
import uuid
from enum import Enum, StrEnum
from typing import override

from qdrant_client import QdrantClient, models
from qdrant_client.models import (
    CollectionInfo,
    Distance,
    Document,
    Fusion,
    FusionQuery,
    Modifier,
    PointStruct,
    Prefetch,
    ScoredPoint,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

from day_three.models.general import (
    ProhibitionTiming,
    RawSubstance,
    TextSubstance,
    WadaCategory,
    WadaStatus,
)
from day_three.vector_db.collectionable import Collectionable
from day_three.vector_db.sparse_point.strategies import (
    Bm25SparsePointGenerator,
    DensePointGenerator,
    SimpleSparsePointGenerator,
    SimpleVectorPointGenerator,
    SparsePointGenerator,
)


class SparseAlgorithms(StrEnum):
    BM25= 'Qdrant/bm25'

class VectorSize(Enum):
    SMALL=384


_WADA_CATEGORY_DESCRIPTIONS = {
    WadaCategory.S1: "Anabolic Agents testosterone steroids nandrolone stanozolol oxandrolone DHEA muscle mass strength protein synthesis androgen receptor",
    WadaCategory.S2: "Peptide Hormones Growth Factors erythropoietin EPO HGH human growth hormone IGF-1 insulin-like CERA darbepoetin GHRP",
    WadaCategory.S3: "Beta-2 Agonists clenbuterol salbutamol formoterol salmeterol bronchodilator asthma inhaler",
    WadaCategory.S4: "Hormone Metabolic Modulators meldonium AICAR GW1516 SARMs selective androgen receptor aromatase inhibitor",
    WadaCategory.S5: "Diuretics Masking Agents furosemide hydrochlorothiazide probenecid plasma expanders epitestosterone",
    WadaCategory.S6: "Stimulants amphetamine cocaine methylphenidate ephedrine modafinil central nervous system",
    WadaCategory.S7: "Narcotics morphine oxycodone fentanyl heroin buprenorphine opioid painkiller",
    WadaCategory.S8: "Cannabinoids cannabis THC marijuana CBD hashish",
    WadaCategory.S9: "Glucocorticoids cortisone prednisone dexamethasone corticosteroid anti-inflammatory",
    WadaCategory.P1: "Beta-Blockers atenolol propranolol metoprolol heart rate anxiety tremor archery shooting",
    WadaCategory.M1: "Blood Doping transfusion autologous homologous blood manipulation oxygen carrier",
    WadaCategory.M2: "Chemical Physical Manipulation catheterization urine substitution adulteration sample tampering",
    WadaCategory.M3: "Gene Cell Doping CRISPR gene editing gene transfer nucleic acid polymer",
    WadaCategory.S0: "Non-Approved Substances experimental unapproved pharmaceutical clinical trial not authorized",
}



def wada_status_from_category(category: WadaCategory | None) -> WadaStatus:
    if category is None:
        return 'legal'
    elif category.timing == ProhibitionTiming.ALL_TIMES:
        return 'banned'
    return 'conditional'

_SIDE_EFFECT_TITLES = ["side effects", "adverse effects", "toxicity", "safety", "health risks"]

class QdrantPreprocessWada(Collectionable):
    """
    Handles qdrant collection management
    """

    def __init__(
        self,
        collection_name: str
    ) -> None:
        """
        Initialize qdrant collection
        """
        self._collection_name = collection_name
        
        self._client = QdrantClient(':memory:')

    def is_created(self) -> bool:
        return self._client.collection_exists(collection_name=self._collection_name)

    @property
    def collection(self):
        """
        Current collection reference
        """
        return self._client.get_collection(self._collection_name)

    def create_collection(
        self, 
    ) -> bool:
        """
        Create in memory collection
        """
        if self.is_created():
            print(f"Collection '{self._collection_name}' already exists")
            self._client.delete_collection(collection_name=self._collection_name)

        success = self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config={},
            sparse_vectors_config={
                "bm25": SparseVectorParams(
                    modifier=Modifier.IDF,
                    index=SparseIndexParams(on_disk=False),
                )
            },
        )

        points = [PointStruct(
            id=idx, 
            vector={
                "bm25": Document(
                    text=_WADA_CATEGORY_DESCRIPTIONS[category],
                    model=SparseAlgorithms.BM25,
                )
            },
            payload={
                'category': str(category)
            }
        ) for idx, category in enumerate(_WADA_CATEGORY_DESCRIPTIONS)]

        update_result = self._client.upsert(self._collection_name, points)

        return success and bool(update_result.operation_id)

    def find_wada(self, substance: TextSubstance) -> WadaCategory | None: 
        """
        find wada status from existing data 
        """
        query = Document(
            text=substance.to_natural_text(),
            model=SparseAlgorithms.BM25
        )

        response = self._client.query_points(
            self._collection_name,
            query=query,
            using="bm25",
            score_threshold=0.2,
        )

        if len(response.points) == 0:
            return None

        most_probable = response.points[0]
        print(f"Most probable wada category is {most_probable}")
        if most_probable.payload is None:
            raise ValueError("Should have payload for the wada points")

        return WadaCategory(most_probable.payload['category'])


    def to_raw_subtance(self, text: TextSubstance) -> RawSubstance:
        wada_category = self.find_wada(text)
        wada_status = wada_status_from_category(wada_category)
        side_effects_cat = next(
            (text.sections[idx] for idx, name in enumerate(text.section_names)
                if name.lower().strip() in _SIDE_EFFECT_TITLES),
            None
        )

        return RawSubstance(
            name=text.name,
            description=text.description,
            drug_category=text.drug_category,
            wada_status=wada_status,
            wada_category=wada_category,
            side_effects=side_effects_cat
        )





class HybridSearchCollection(Collectionable):
    def __init__(
        self,
        collection_name: str,
        sparse_generator: SparsePointGenerator | None = None,
        dense_generator: DensePointGenerator | None = None,
        url: str | None = None,
        api_key: str | None = None
    ) -> None:
        super().__init__()

        self._collection_name = collection_name
        self._url = url or os.getenv('QDRANT_URL')
        self._api_key = api_key or os.getenv('QDRANT_API_KEY')

        if not self._url or not self._api_key:
            raise ValueError("Qdrant URL and API key must be provided")

        if sparse_generator is None:
            self._sparse_generator = Bm25SparsePointGenerator()
        else:
            self._sparse_generator = sparse_generator
        
        if dense_generator is None:
            self._dense_generator = SimpleVectorPointGenerator()
        else:
            self._dense_generator = dense_generator

        self._client = QdrantClient(url=self._url, api_key=self._api_key)
        
    @override
    def create_collection(self) -> bool:
        created = self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config={
                'dense': VectorParams(size=VectorSize.SMALL.value, distance=Distance.COSINE)
            },
            sparse_vectors_config={
                'sparse': SparseVectorParams(
                    modifier=Modifier.IDF,
                    index=SparseIndexParams(on_disk=False)
                )
            }
        )

        return created

    @override
    def collection(self) -> CollectionInfo:
        return self._client.get_collection(self._collection_name)


    def add_data_points(self, substrances: list[RawSubstance]):
        """
        Add substances as data points
        """

        points: list[PointStruct] = []

        for substance in substrances:
            dense_vector = self._dense_generator.generate_point(substance.to_natural_text())
            sparse_vector = self._sparse_generator.generate_point(substance.to_natural_text())

            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),  
                    vector={
                        'dense': dense_vector,
                        'sparse': sparse_vector
                    },
                    payload=substance.model_dump()
                )
            )
        
        self._client.upload_points(self._collection_name, points)

    
    def hybrid_search(self, text: str, query_limit: int = 10, prefetch_limit: int = 20) -> list[ScoredPoint]:
        """
        Perform a hybrid search over the data
        """

        query_dense = self._dense_generator.generate_point(text)
        query_sparse = self._sparse_generator.generate_point(text)

        response = self._client.query_points(
            self._collection_name,
            prefetch=[
                Prefetch(
                    query=query_dense,
                    using='dense',
                    limit=prefetch_limit
                ),
                Prefetch(
                    query=query_sparse,
                    using='sparse',
                    limit=prefetch_limit
                )
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=query_limit
        )

        return response.points


    def dense_search(self, text: str, query_limit = 10) -> list[ScoredPoint]:
        """
        Retrieve using only dense vectors
        """

        query_dense = self._dense_generator.generate_point(text)
        result = self._client.query_points(
            self._collection_name,
            query=query_dense,
            using='dense',
            limit=query_limit
        )

        return result.points

    def sparse_search(self, text: str, query_limit = 10) -> list[ScoredPoint]:
        """
        Retrieve using only sparse vectors
        """
        query_sparse = self._sparse_generator.generate_point(text)
        result = self._client.query_points(
            self._collection_name,
            query=query_sparse,
            using='sparse',
            limit=query_limit,
        )

        return result.points



            
    