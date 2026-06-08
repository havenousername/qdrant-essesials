"""
Point building functionality for movie data
"""

import json
import pickle
import uuid
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image
from qdrant_client import models
from sentence_transformers import SentenceTransformer

from models.movie_models import RawMovie

from .chunking import fixed_size_chunks, semantic_chunks, sentence_chunks

# Initialize encoders for different strategies
encoders = {
    'fixed': SentenceTransformer('all-MiniLM-L6-v2'),
    # 'sentence': SentenceTransformer("BAAI/bge-base-en-v1.5"),
    'semantic': SentenceTransformer("BAAI/bge-base-en-v1.5"),
    'poster': SentenceTransformer("clip-ViT-B-32"),
}

# Vector configurations for each strategy
vector_configs: Mapping[str, models.VectorParams] = {
    'fixed': models.VectorParams(size=384, distance=models.Distance.COSINE),
    # 'sentence': models.VectorParams(size=768, distance=models.Distance.COSINE),
    'semantic': models.VectorParams(size=768, distance=models.Distance.COSINE),
    'poster': models.VectorParams(size=512, distance=models.Distance.COSINE),
}


class PointBuilder:
    """Builds Qdrant points from movie data with text chunks and image embeddings"""

    def __init__(self):
        self.encoders = encoders
        self.chunking_strategies = {
            "fixed": fixed_size_chunks,
            "sentence": sentence_chunks,
            "semantic": semantic_chunks,
        }

    def load_image_from_url(self, url: str) -> Image.Image | None:
        """
        Safely load image from URL

        Args:
            url: Image URL

        Returns:
            PIL Image object or None if loading fails
        """
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        except Exception as e:
            print(f"Failed to load image from {url}: {e}")
            return None

    def build_points_from_movies(
        self,
        movies: list[RawMovie],
        include_images: bool = True,
        image_base_url: str = "https://image.tmdb.org/t/p/w500/",
        batch_size: int = 32,
        max_workers: int = 4,
    ) -> list[models.PointStruct]:
        """
        Build points from a list of movies with parallel processing

        Args:
            movies: List of RawMovie objects
            include_images: Whether to include image embeddings
            image_base_url: Base URL for movie poster images
            batch_size: Number of texts to encode at once
            max_workers: Number of parallel workers for image downloading

        Returns:
            List of PointStruct objects ready for upload
        """
        print(f"Building points for {len(movies)} movies with batch_size={batch_size}, max_workers={max_workers}")
        
        # Pre-download all images in parallel if needed
        image_vectors = {}
        if include_images:
            print("Downloading and encoding movie posters in parallel...")
            image_vectors = self._batch_download_and_encode_images(
                movies, image_base_url, max_workers
            )
        
        # Batch encode all texts
        print("Encoding text chunks in batches...")
        points = self._batch_build_points(
            movies, image_vectors, batch_size
        )
        
        return points

    def build_points_from_movie(
        self,
        movie: RawMovie,
        include_images: bool = True,
        image_base_url: str = "https://image.tmdb.org/t/p/w500/",
    ) -> list[models.PointStruct]:
        """
        Build points from a single movie

        Args:
            movie: RawMovie object
            include_images: Whether to include image embeddings
            image_base_url: Base URL for movie poster images

        Returns:
            List of PointStruct objects for this movie
        """
        points = []

        # Get movie text content
        plot = movie.plot if movie.plot else movie.overview
        if not plot:
            print(f"Warning: No plot/overview for movie {movie.title}")
            return points

        # Load and encode image if requested
        image_vector: list[float] | None = None
        if include_images and movie.poster_path:
            image_url = f"{image_base_url}{movie.poster_path}"
            image = self.load_image_from_url(image_url)
            if image:
                image_vector = self.encoders["poster"].encode(image).tolist()

        # Process with each chunking strategy
        for strategy_name, chunking_fn in self.chunking_strategies.items():
            chunks = chunking_fn(plot)

            for chunk_idx, chunk in enumerate(chunks):
                # Create vectors for this chunk
                vectors: dict[str, Any] = {}
                
                # Only add vector if encoder exists for this strategy
                if strategy_name in self.encoders:
                    vectors[strategy_name] = self.encoders[strategy_name].encode(chunk).tolist()

                # Add image vector if available
                if image_vector:
                    vectors["poster"] = image_vector
                
                # Skip if no vectors were created
                if not vectors:
                    print(f"Warning: No vectors created for {strategy_name} chunk {chunk_idx} of {movie.title}")
                    continue

                # Create point
                point = models.PointStruct(
                    id=str(uuid.uuid4()),  # Ensure string ID
                    vector=vectors,
                    payload={
                        **movie.model_dump(),  # Include all movie metadata
                        "chunk": chunk,
                        "chunk_strategy": strategy_name,
                        "chunk_index": chunk_idx,
                        "has_image": image_vector is not None,
                        "poster_url": f"{image_base_url}{movie.poster_path}"
                        if movie.poster_path
                        else None,
                    },
                )

                points.append(point)

        return points
    
    def _batch_download_and_encode_images(
        self,
        movies: list[RawMovie],
        image_base_url: str,
        max_workers: int = 4
    ) -> dict[int, list[float]]:
        """Download and encode images in parallel"""
        image_vectors = {}
        
        def download_and_encode(idx: int, movie: RawMovie):
            if not movie.poster_path:
                return idx, None
            
            image_url = f"{image_base_url}{movie.poster_path}"
            image = self.load_image_from_url(image_url)
            
            if image:
                # Encode single image
                vector = self.encoders["poster"].encode(image).tolist()
                return idx, vector
            return idx, None
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(download_and_encode, idx, movie): idx
                for idx, movie in enumerate(movies)
            }
            
            completed = 0
            for future in as_completed(futures):
                idx, vector = future.result()
                if vector:
                    image_vectors[idx] = vector
                completed += 1
                if completed % 50 == 0:
                    print(f"  Processed {completed}/{len(movies)} images")
        
        print(f"Successfully encoded {len(image_vectors)} images")
        return image_vectors
    
    def _batch_build_points(
        self,
        movies: list[RawMovie],
        image_vectors: dict[int, list[float]],
        batch_size: int = 32
    ) -> list[models.PointStruct]:
        """Build points with batch text encoding"""
        all_points = []
        
        # Collect all texts to encode per strategy
        texts_by_strategy = {strategy: [] for strategy in self.chunking_strategies}
        text_metadata = []  # Track which text belongs to which movie/chunk
        
        # First pass: collect all texts
        for movie_idx, movie in enumerate(movies):
            plot = movie.plot if movie.plot else movie.overview
            if not plot:
                print(f"Warning: No plot/overview for movie {movie.title}")
                continue
            
            for strategy_name, chunking_fn in self.chunking_strategies.items():
                chunks = chunking_fn(plot)
                
                for chunk_idx, chunk in enumerate(chunks):
                    texts_by_strategy[strategy_name].append(chunk)
                    text_metadata.append({
                        'movie_idx': movie_idx,
                        'movie': movie,
                        'chunk_idx': chunk_idx,
                        'chunk': chunk,
                        'strategy': strategy_name
                    })
        
        # Second pass: batch encode texts per strategy
        encoded_vectors = {}
        for strategy_name, texts in texts_by_strategy.items():
            if strategy_name in self.encoders and texts:
                print(f"  Encoding {len(texts)} {strategy_name} chunks in batches...")
                
                # Encode in batches
                all_vectors = []
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    # encode_multi returns numpy array
                    batch_vectors = self.encoders[strategy_name].encode(
                        batch_texts, 
                        batch_size=batch_size,
                        show_progress_bar=False
                    )
                    all_vectors.extend(batch_vectors.tolist())
                
                encoded_vectors[strategy_name] = all_vectors
        
        # Third pass: create points with pre-encoded vectors
        strategy_counters = {strategy: 0 for strategy in self.chunking_strategies}
        
        for metadata in text_metadata:
            movie_idx = metadata['movie_idx']
            movie = metadata['movie']
            chunk_idx = metadata['chunk_idx']
            chunk = metadata['chunk']
            strategy = metadata['strategy']
            
            vectors = {}
            
            # Add text vector
            if strategy in encoded_vectors:
                vector_idx = strategy_counters[strategy]
                vectors[strategy] = encoded_vectors[strategy][vector_idx]
                strategy_counters[strategy] += 1
            
            # Add image vector if available
            if movie_idx in image_vectors:
                vectors["poster"] = image_vectors[movie_idx]
            
            if not vectors:
                continue
            
            # Create point
            point = models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vectors,
                payload={
                    **movie.model_dump(),
                    "chunk": chunk,
                    "chunk_strategy": strategy,
                    "chunk_index": chunk_idx,
                    "has_image": movie_idx in image_vectors,
                    "poster_url": f"https://image.tmdb.org/t/p/w500/{movie.poster_path}"
                    if movie.poster_path else None,
                },
            )
            all_points.append(point)
        
        print(f"Created {len(all_points)} points total")
        return all_points

    def save_points_to_disk(self, points: list[models.PointStruct], filepath: str | Path) -> None:
        """
        Save points to disk for later use
        
        Args:
            points: List of PointStruct objects
            filepath: Path to save the points
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving {len(points)} points to {filepath}")
        
        # Convert points to serializable format
        serializable_points = []
        for point in points:
            # Convert PointStruct to dict
            point_dict = {
                'id': point.id,
                'vector': point.vector,
                'payload': point.payload
            }
            serializable_points.append(point_dict)
        
        # Save as pickle for efficiency with large datasets
        with open(filepath, 'wb') as f:
            pickle.dump(serializable_points, f)
        
        print(f"Successfully saved points to {filepath}")
        
    def load_points_from_disk(self, filepath: str | Path) -> list[models.PointStruct]:
        """
        Load previously saved points from disk
        
        Args:
            filepath: Path to load the points from
            
        Returns:
            List of PointStruct objects
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Points file not found: {filepath}")
            
        print(f"Loading points from {filepath}")
        
        with open(filepath, 'rb') as f:
            serializable_points = pickle.load(f)
        
        # Convert back to PointStruct objects
        points = []
        for point_dict in serializable_points:
            point = models.PointStruct(
                id=point_dict['id'],
                vector=point_dict['vector'],
                payload=point_dict['payload']
            )
            points.append(point)
        
        print(f"Successfully loaded {len(points)} points from {filepath}")
        return points
        
    def save_points_as_json(self, points: list[models.PointStruct], filepath: str | Path) -> None:
        """
        Save points in human-readable JSON format (useful for debugging)
        
        Args:
            points: List of PointStruct objects
            filepath: Path to save the points
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving {len(points)} points to {filepath} as JSON")
        
        # Convert points to serializable format
        serializable_points = []
        for point in points:
            # Extract only basic info for readability
            point_info = {
                'id': point.id,
                'title': point.payload.get('title', 'Unknown'),
                'chunk_strategy': point.payload.get('chunk_strategy'),
                'chunk_index': point.payload.get('chunk_index'),
                'has_image': point.payload.get('has_image'),
                'vector_keys': list(point.vector.keys()) if isinstance(point.vector, dict) else ['single_vector']
            }
            serializable_points.append(point_info)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_points, f, indent=2)
            
        print(f"Successfully saved points summary to {filepath}")