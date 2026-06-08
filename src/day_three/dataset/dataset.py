

import pickle
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import cast, override

import wikipediaapi
from pydantic import BaseModel

from day_three.models.general import DataChunk, RawSubstance, TextSubstance
from day_three.vector_db.qdrant_wrapper import QdrantPreprocessWada


class AbstractDataset(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def location(self) -> str:
        ...


class WikiCashedDataset[Type: BaseModel](AbstractDataset):
    def __init__(
        self,
        categories: list[str],
        cache_dir: str = '.cache/dataset',
        chunk_size: int = 1000,
    ) -> None:
        super().__init__()
        self._cache_dir: Path = Path(cache_dir)
        self._chunk_size: int = chunk_size
        self._search_categories = categories
        assert len(self._search_categories) > 0
        # current category pointer
        self._current_category_idx: int = 0
        # cursor for the current chunk 
        self._chunk_cursor: int = 0

        self._wiki = wikipediaapi.Wikipedia('PEDSearch/1.0', "en")
        self._queries_members: list[wikipediaapi.WikipediaPage] = []

        self._buffer: dict[str, list[DataChunk[Type]]] = {}
        # Categories whose full member list has already been pulled from Wikipedia.
        # Lets us short-circuit re-fetches at category boundaries.
        self._fully_fetched: set[str] = set()

    @property
    def current_category(self) -> str | None:
        try: 
            return self._search_categories[self._current_category_idx]
        except IndexError:
            return None

    def go_to_next_category(self) -> None:
        self._current_category_idx += 1

    def _cache_file(self, category: str, chunk_idx: int) -> Path:
        slug = category.replace(" ", "_")

        return self._cache_dir / slug / f'chunk_{chunk_idx}.pkl'

    @abstractmethod
    def _transform(self, page: wikipediaapi.WikipediaPage, category: str) -> Type:
        """Convert a WikipediaPage into an instance of T."""
        ...

    def _fetch_category(self, category: str) -> list[Type]:
        """
        Fetch elements of a specific category 
        """
        category_page = self._wiki.page(category)

        # PagesDict is typed as dict[str, BaseWikipediaPage] but always holds
        # WikipediaPage at runtime; cast because BaseWikipediaPage is not public.
        pages = cast(
            list[wikipediaapi.WikipediaPage], 
            list(category_page.categorymembers.values())
        )

        self._queries_members.extend(pages)
        local_t: list[Type] = []

        for page in pages:
            local_t.append(self._transform(page, category))

        
        return local_t

    def _load_local_chunk(self, category: str, chunk_idx: int) -> list[Type] | None:
        """
        Returns from the local files categories of data
        """
        with open(self._cache_file(category, chunk_idx), 'rb') as f:
            data: list[Type] = pickle.load(f)
            return data

    
    def _upload_local_chunk(self, category: str, chunk_idx: int, data: list[Type]) -> None:
        """
        Uploads chunk into the filesystem
        """
        path = self._cache_file(category, chunk_idx)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(data, f)


    def _fetch_or_load(self, chunk_idx: int) -> DataChunk[Type] | None:
        """
        Returns chunk_idx for the current category from buffer, disk cache, or
        a fresh Wikipedia fetch. Returns None when the category has no chunk at
        that index. Does not mutate self._buffer — that's _advance's job.
        """

        if self.current_category is None:
            return None

        chunks = self._buffer.get(self.current_category, [])
        if chunk_idx < len(chunks):
            return chunks[chunk_idx]

        if self._cache_file(self.current_category, chunk_idx).exists():
            data = self._load_local_chunk(self.current_category, chunk_idx) or []
            return DataChunk(chunk=data, index=chunk_idx)

        # Not in buffer, not on disk. If we've already fetched this category in
        # this session, we know there's nothing more — skip the redundant call.
        if self.current_category in self._fully_fetched:
            return None

        categories: list[Type] = self._fetch_category(self.current_category)

        categories_in_chunks: list[list[Type]] = [
            categories[i:i + self._chunk_size] for i in range(0, len(categories), self._chunk_size)
        ]

        for idx, category in enumerate(categories_in_chunks):
            self._upload_local_chunk(self.current_category, idx, category)

        self._fully_fetched.add(self.current_category)

        return (
            DataChunk(chunk=categories_in_chunks[chunk_idx], index=chunk_idx)
            if chunk_idx < len(categories_in_chunks)
            else None
        )
    
    def __iter__(self) -> Iterator[Type]:
        self._current_category_idx = 0
        self._chunk_cursor = 0
        self._buffer = {}
        return self

    def __next__(self) -> Type:
        while True:
            if self.current_category is None:
                raise StopIteration
            chunks = self._buffer.get(self.current_category, [])
            if self._chunk_cursor >= len(chunks):
                self._advance()
                continue

            chunk = chunks[self._chunk_cursor]
            if chunk.position >= len(chunk.chunk):
                self._chunk_cursor += 1
                continue
            
            item = chunk.chunk[chunk.position]
            chunk.position += 1
            return item

    def _advance(self) -> None:
        if self.current_category is None:
            raise StopIteration
        
        chunk = self._fetch_or_load(self._chunk_cursor)
        if chunk is None:
            self.go_to_next_category()
            self._chunk_cursor = 0
            if self._current_category_idx >= len(self._search_categories):
                raise StopIteration
            return
        
        self._buffer.setdefault(self.current_category, [])
        self._buffer[self.current_category].append(chunk)


    @property
    @override
    def name(self) -> str:
        return 'WIKI_Dataset'

    @property
    @override
    def location(self) -> str:
        return 'wikipedia'



class DrugSearchDataset(WikiCashedDataset[TextSubstance]):
    def __init__(self, categories: list[str]) -> None:
        super().__init__(categories)

    @override
    def _transform(self, page: wikipediaapi.WikipediaPage, category: str) -> TextSubstance:
        """
        Transform from the wikipedia page to a streamlined text subtance
        """
        return TextSubstance(
            name=page.title,
            description=page.summary,
            drug_category=category,
            sections=[section.full_text() for section in page.sections],
            section_names=[section.title for section in page.sections]
        )
        

class DrugRawSubtranceDataset(WikiCashedDataset[RawSubstance]):
    def __init__(self, categories: list[str], cache_dir: str = '.cache/dataset', chunk_size: int = 1000) -> None:
        super().__init__(categories)
        self._qdrant_preprocess = QdrantPreprocessWada('wada-preprocess')
        self._qdrant_preprocess.create_collection()

    @override
    def _transform(self, page: wikipediaapi.WikipediaPage, category: str) -> RawSubstance:
        """
        Transform from the wikipedia page to a streamlined raw subtance
        """
        text_subtrance = TextSubstance(
            name=page.title,
            description=page.summary,
            drug_category=category,
            sections=[section.full_text() for section in page.sections],
            section_names=[section.title for section in page.sections]
        )

        return self._qdrant_preprocess.to_raw_subtance(text_subtrance)



    