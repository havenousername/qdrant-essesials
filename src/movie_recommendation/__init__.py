"""
Movie Recommendation System using Qdrant Vector Database

This package provides functionality for building a semantic search engine
for movies using multiple text chunking strategies and multimodal search
with image embeddings.
"""

from .dataset import MoviesDataset
from .search import MovieSearch
from .point_builder import PointBuilder
from .chunking import fixed_size_chunks, sentence_chunks, semantic_chunks
from .visualization import (
    display_movie_result,
    display_search_results,
    display_search_summary,
    display_unique_movies
)

__all__ = [
    # Dataset
    'MoviesDataset',
    
    # Search
    'MovieSearch',
    
    # Point Building
    'PointBuilder',
    
    # Chunking
    'fixed_size_chunks',
    'sentence_chunks', 
    'semantic_chunks',
    
    # Visualization
    'display_movie_result',
    'display_search_results',
    'display_search_summary',
    'display_unique_movies',
]