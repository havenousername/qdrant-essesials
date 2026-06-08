


from abc import ABC, abstractmethod

from qdrant_client.models import CollectionInfo


class Collectionable(ABC):
    def __init__(self) -> None:
        super().__init__()


    @abstractmethod
    def create_collection(self) -> bool:
        """
        Create current instance of collection
        """

    @property
    @abstractmethod
    def collection(self) -> CollectionInfo:
        """
        Current target collection 
        """