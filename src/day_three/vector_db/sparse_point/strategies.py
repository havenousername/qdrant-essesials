

from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, override

import regex as re
from qdrant_client.models import Document, SparseVector
from sentence_transformers import SentenceTransformer

_global_vocabulary: dict[str, int] = {}


class SparsePointGenerator(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def generate_point(self, text: str) -> SparseVector | Document:
        """
        Generate a new sparse vector
        """


class DensePointGenerator(ABC):
    def __init__(self, size: int) -> None:
        super().__init__()
        self._vector_size = size

    @abstractmethod
    def generate_point(self, text: str) -> list[Any]:
        """
        Generate a new dense point
        """

class SimpleSparsePointGenerator(SparsePointGenerator):
    def __init__(self) -> None:
        super().__init__()

    @override
    def generate_point(self, text: str) -> SparseVector:
        # use simple tokenization: each token is a word
        words: list[str] = re.findall(r'\b\w+\b', text.lower())
        words_counter = Counter(words)

        indices: list[int] = []
        values: list[float] = []

        for word, counter in words_counter.items(): 
            if word not in _global_vocabulary:
                _global_vocabulary[word] = len(_global_vocabulary)

            indices.append(_global_vocabulary[word])
            values.append(float(counter))
        return SparseVector(indices=indices, values=values)



class Bm25SparsePointGenerator(SparsePointGenerator):
    _MODEL = 'Qdrant/bm25'

    def __init__(self) -> None:
        super().__init__()

    @override
    def generate_point(self, text: str) -> Document:
        return Document(text=text, model=self._MODEL)


class SimpleVectorPointGenerator(DensePointGenerator):
    def __init__(self, size: int = 384) -> None:
        super().__init__(size)
        self._encoder = SentenceTransformer("all-MiniLM-L6-v2")

    def generate_point(self, text: str) -> list:
        return self._encoder.encode(text).tolist()

