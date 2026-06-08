"""
Movie search functionality using Qdrant vector database
"""
import os
import statistics
import time
from collections.abc import Mapping
from typing import Any, final

import numpy as np
from pydantic import BaseModel
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer


class HNSWConfig(BaseModel):
    name: str # name of the config
    connection_per_node: int  # higher -> better and more storage
    built_graph_accuracy: int # higher -> better and slower


@final
class MovieSearch:
    """Handles movie search operations using Qdrant vector database"""
    
    def __init__(self, 
        encoders: dict[str, SentenceTransformer],
        vector_configs: Mapping[str, models.VectorParams],
        collection_name: str = "movie_recommendation", 
        hnsw_config: HNSWConfig | None = None,
        url: str | None = None, 
        api_key: str | None = None):
        """
        Initialize MovieSearch with Qdrant client
        
        Args:
            collection_name: Name of the Qdrant collection
            url: Qdrant URL (defaults to env var QDRANT_URL)
            api_key: Qdrant API key (defaults to env var QDRANT_API_KEY)
        """
        self.collection_name = collection_name
        self.url = url or os.getenv('QDRANT_URL')
        self.api_key = api_key or os.getenv('QDRANT_API_KEY')
        
        if not self.url or not self.api_key:
            raise ValueError("Qdrant URL and API key must be provided")
        
        self.client = QdrantClient(url=self.url, api_key=self.api_key)
        self.encoders = encoders
        self.vector_configs = vector_configs
        self.hnsw_config = hnsw_config

    def create_collection(self):
        """Create collection with configured vector parameters"""
        if self.client.collection_exists(collection_name=self.collection_name):
            print(f"Collection '{self.collection_name}' already exists")
            self.client.delete_collection(collection_name=self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=self.vector_configs,
            hnsw_config=models.HnswConfigDiff(
                m=self.hnsw_config.built_graph_accuracy,
                ef_construct=self.hnsw_config.built_graph_accuracy,
                full_scan_threshold=10 # force HNSW instead of the full scan
            ) if self.hnsw_config else None,
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=10 # force indexing even on small sets of demo
            )
        )
        
        # Create payload index for filtering
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="chunk_strategy",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        
        print(f"Created collection '{self.collection_name}'")
        return True
    
    def upload_points(self, points: list[models.PointStruct], batch_size: int = 100):
        """
        Upload points to collection in batches
        
        Args:
            points: List of PointStruct objects to upload
            batch_size: Size of upload batches
        """
        total_points = len(points)

        start_time = time.time()
        for i in range(0, total_points, batch_size):
            batch = points[i:i+batch_size]
            self.client.upload_points(
                collection_name=self.collection_name,
                points=batch,
                wait=True
            )
            print(f"Uploaded batch {i//batch_size + 1}/{(total_points + batch_size - 1)//batch_size}")
        
        end_time = time.time() - start_time
        print(f"{self.hnsw_config.name if self.hnsw_config else None}: Uploaded {len(points)} points in {end_time:.2f}s")
        print(f"Total points uploaded: {total_points}")
        return end_time

    def average_score(self, scores: list[models.ScoredPoint]) -> float:
        """Calculate average score from list of scored points"""
        if not scores:
            return 0.0
        return statistics.mean([point.score for point in scores])
    
    def search_results(self, query: str, image_query: Any | None = None, 
                    weights: dict[str, float] = {'image': 0.3, 'text': 0.7},
                    limit: int = 20) -> list[dict[str, Any]]:
        """
        Search using text strategies and optional image query with fusion
        
        Args:
            query: Text search query
            image_query: Optional image for multimodal search
            weights: Weights for combining text and image scores
            limit: Maximum number of results
            
        Returns:
            List of search results with combined scores
        """

        print(f"Searching for: {query}")
        
        strategies = ['fixed', 'sentence', 'semantic']
        text_scores: dict[str, float] = {}
        text_results: dict[str, list[models.ScoredPoint]] = {}
        
        # Search with each text strategy
        for strategy in strategies:
            result = self.client.query_points(
                collection_name=self.collection_name,
                query=self.encoders[strategy].encode(query).tolist(),
                using=strategy,
                with_payload=True,
                limit=limit
            )
            
            text_scores[strategy] = self.average_score(result.points)
            text_results[strategy] = result.points
            
            print(f'\n--- {strategy.upper()} STRATEGY ---')
            
            if not result.points:
                print("   No results found")
                continue
            
            for i, point in enumerate(result.points[:5]):  # Show top 5
                if point.payload:
                    print(f'{i+1}. {point.payload.get("title", "Unknown")}')
                    print(f'   Score: {point.score:.3f}')
                    chunk = point.payload.get('chunk', '')[:80]
                    print(f'   Chunk: {chunk}...')
                    print()
        
        # Find best text strategy
        if not text_scores:
            print("No results found for any strategy")
            return []
        
        best_strategy = max(text_scores.items(), key=lambda x: x[1])
        print(f"\nBest text strategy: {best_strategy[0]} (avg score: {best_strategy[1]:.3f})")
        best_results = text_results[best_strategy[0]]
        
        # Apply image reranking if provided
        if image_query and best_results:
            image_vector = self.encoders['poster'].encode(image_query).tolist()
            
            # Get unique movie IDs
            movie_ids = set()
            for result in best_results:
                if result.payload:
                    movie_ids.add(result.payload.get('id'))
            
            # Query with image vector for these movies
            image_results = self.client.query_points(
                collection_name=self.collection_name,
                query=image_vector,
                using='poster',
                filter=models.Filter(
                    must=[
                        models.HasIdCondition(has_id=list(movie_ids))
                    ]
                )
            )
            
            # Create image score lookup
            image_score_map = {p.id: p.score for p in image_results.points}
            
            # Combine scores
            final_results = []
            for point in best_results:
                text_score = point.score
                image_score = image_score_map.get(point.id, 0.0)
                combined_score = weights['text'] * text_score + weights['image'] * image_score
                
                final_results.append({
                    'point': point,
                    'score': combined_score,
                    'text_score': text_score,
                    'image_score': image_score
                })
            
            # Sort by combined score
            return sorted(final_results, key=lambda x: x['score'], reverse=True)
        
        # Return text-only results
        return [{'point': p, 'score': p.score, 'text_score': p.score, 'image_score': 0.0} 
                for p in best_results]
    
    def search_unique_movies(self, query: str, limit: int = 10) -> list[models.ScoredPoint]:
        """
        Search for movies ensuring unique results (no duplicate movies)
        
        Args:
            query: Search query
            limit: Maximum number of unique movies to return
            
        Returns:
            List of unique movie results
        """
        result = self.client.query_points(
            collection_name=self.collection_name,
            query=self.encoders['semantic'].encode(query).tolist(),
            using='semantic',
            limit=limit * 5  # Get more to ensure variety
        )
        
        seen_titles = set()
        unique_results = []
        
        for point in result.points:
            if point.payload:
                title = point.payload.get('title')
                if title and title not in seen_titles:
                    seen_titles.add(title)
                    unique_results.append(point)
                    if len(unique_results) >= limit:
                        break
        
        return unique_results

    def benchmark_search(self, query: str | list[str], ef_values = [64, 128, 256]):
        import random
        # Pre-encode query to isolate search time from encoding time
        query_vector = self.encoders['semantic'].encode(query if isinstance(query, str) else query[0]).tolist()
        
        # Warmup query
        self.client.query_points(
            collection_name=self.collection_name, 
            query=query_vector,
            using='semantic',
            limit=1
        )
        
        # Check if HNSW is actually being used
        collection_info = self.client.get_collection(self.collection_name)
        print(f"Collection info: {collection_info.config.hnsw_config}")
        print(f"Total points: {collection_info.points_count}")
        print(f"Indexed vectors: {collection_info.indexed_vectors_count}")

        results: dict[int, Any] = {}
        for hnsw_ef in ef_values:
            times = []

            print(f"\n===Start search for hnsw_ef={hnsw_ef}===")

            for i in range(25):
                start_time = time.time()
                result = self.client.query_points(
                    collection_name=self.collection_name, 
                    query=query_vector,
                    using='semantic',
                    limit=10,
                    search_params=models.SearchParams(
                        hnsw_ef=hnsw_ef
                    ),
                    with_payload=False
                )
                search_time = (time.time() - start_time) * 1000
                times.append(search_time)
                
                # update vector search query 
                index = 0 if isinstance(query, str) else random.randrange(0, len(query))
                query_vector = self.encoders['semantic'].encode(query if isinstance(query, str) else query[index]).tolist()
                # Print every 5th iteration to reduce noise
                if i % 5 == 0:
                    print(f"Search {i+1}/25: {search_time:.2f}ms (found {len(result.points)} results)")

            results[hnsw_ef] = {
                'avg_time': np.mean(times),
                'min_time': np.min(times),
                'max_time': np.max(times),
                'std_time': np.std(times)
            }
            print(f"hnsw_ef={hnsw_ef}: avg={results[hnsw_ef]['avg_time']:.2f}ms ±{results[hnsw_ef]['std_time']:.2f}ms")
            
        return results
    
    def benchmark_brute_force_vs_hnsw(self, query: str):
        """Compare brute force vs HNSW performance"""
        query_vector = self.encoders['semantic'].encode(query).tolist()
        
        print("\n=== Brute Force vs HNSW Comparison ===")
        
        # Test brute force (exact search)
        brute_times = []
        for i in range(10):
            start_time = time.time()
            result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                using='semantic',
                limit=10,
                search_params=models.SearchParams(exact=True),  # Force exact search
                with_payload=False
            )
            brute_times.append((time.time() - start_time) * 1000)
            if i % 2 == 0:
                print(f"Brute force {i+1}/10: {brute_times[-1]:.2f}ms")
        
        # Test HNSW
        hnsw_times = []
        for i in range(10):
            start_time = time.time()
            result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                using='semantic',
                limit=10,
                search_params=models.SearchParams(hnsw_ef=128),  # Use HNSW
                with_payload=False
            )
            hnsw_times.append((time.time() - start_time) * 1000)
            if i % 2 == 0:
                print(f"HNSW {i+1}/10: {hnsw_times[-1]:.2f}ms")
        
        brute_avg = np.mean(brute_times)
        hnsw_avg = np.mean(hnsw_times)
        speedup = brute_avg / hnsw_avg if hnsw_avg > 0 else 1
        
        print(f"\nResults:")
        print(f"Brute Force: {brute_avg:.2f}ms ±{np.std(brute_times):.2f}ms")
        print(f"HNSW:        {hnsw_avg:.2f}ms ±{np.std(hnsw_times):.2f}ms")
        print(f"Speedup:     {speedup:.2f}x")
        
        return {
            'brute_force': {'avg': brute_avg, 'std': np.std(brute_times)},
            'hnsw': {'avg': hnsw_avg, 'std': np.std(hnsw_times)},
            'speedup': speedup
        }

    def test_hnsw_vs_exact(self, query: str, num_tests: int = 10):
        """Test if HNSW is actually being used by comparing with forced exact search"""
        query_vector = self.encoders['semantic'].encode(query).tolist()
        
        print(f"\n=== Testing HNSW vs Exact Search ===")
        
        # Get collection info
        info = self.client.get_collection(self.collection_name)
        hnsw_config = info.config.hnsw_config
        
        print(f"Collection: {self.collection_name}")
        print(f"Points: {info.points_count}")
        print(f"Indexed vectors: {info.indexed_vectors_count}")
        print(f"HNSW m: {hnsw_config.m}")
        print(f"HNSW ef_construct: {hnsw_config.ef_construct}")
        print(f"Full scan threshold: {hnsw_config.full_scan_threshold}")
        
        # Test exact search
        exact_times = []
        print(f"\nTesting exact search...")
        for i in range(num_tests):
            start = time.time()
            exact_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                using='semantic',
                limit=10,
                search_params=models.SearchParams(exact=True),
                with_payload=False
            )
            exact_times.append((time.time() - start) * 1000)
            
        # Test HNSW search
        hnsw_times = []
        print(f"Testing HNSW search...")
        for i in range(num_tests):
            start = time.time()
            hnsw_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                using='semantic',
                limit=10,
                search_params=models.SearchParams(hnsw_ef=128),
                with_payload=False
            )
            hnsw_times.append((time.time() - start) * 1000)
        
        # Compare results
        exact_avg = np.mean(exact_times)
        hnsw_avg = np.mean(hnsw_times)
        
        print(f"\nResults:")
        print(f"Exact search: {exact_avg:.2f}ms ±{np.std(exact_times):.2f}")
        print(f"HNSW search:  {hnsw_avg:.2f}ms ±{np.std(hnsw_times):.2f}")
        print(f"Speedup:      {exact_avg/hnsw_avg:.2f}x")
        
        # Check if results are identical (they should be very similar)
        exact_ids = [p.id for p in exact_result.points]
        hnsw_ids = [p.id for p in hnsw_result.points]
        
        overlap = len(set(exact_ids) & set(hnsw_ids))
        print(f"Result overlap: {overlap}/{len(exact_ids)} ({100*overlap/len(exact_ids):.1f}%)")
        
        # Determine if HNSW is working
        if (info.points_count or 0) < hnsw_config.full_scan_threshold:
            print(f"\n⚠️  HNSW likely NOT used: points ({info.points_count}) < threshold ({hnsw_config.full_scan_threshold})")
        elif exact_avg / hnsw_avg < 1.1:  # Less than 10% difference
            print(f"\n⚠️  HNSW might not be used: minimal performance difference")
        else:
            print(f"\n✅ HNSW appears to be working: {exact_avg/hnsw_avg:.2f}x speedup")
        
        return {
            'exact_avg': exact_avg,
            'hnsw_avg': hnsw_avg,
            'speedup': exact_avg / hnsw_avg,
            'overlap_percent': 100 * overlap / len(exact_ids),
            'points_count': info.points_count,
            'full_scan_threshold': hnsw_config.full_scan_threshold
        }

    def wait_for_indexing(self, timeout = 120, poll_interval = 1):
        print(f"Waiting for the collection {self.collection_name} to be indexed...")


        start_time = time.time()

        while time.time() - start_time < timeout:
            info = self.client.get_collection(self.collection_name)

            if (info.indexed_vectors_count or 0) > 0 and info.status == models.CollectionStatus.GREEN:
                print(f"Success! Collection '{self.collection_name}' is indexed and ready.")
                print(f" - Status: {info.status.value}")
                print(f" - Indexed vectors: {info.indexed_vectors_count}")
                return
            print(f" - Status: {info.status.value}, Indexed vectors: {info.indexed_vectors_count}. Waiting...")
            time.sleep(poll_interval)

        info = self.client.get_collection(self.collection_name)
        raise RuntimeError(
            f"Timeout reached after {timeout} seconds. Collection '{self.collection_name}' is not ready. "
            f"Final status: {info.status.value}, Indexed vectors: {info.indexed_vectors_count}"
        )



    def benchmark_with_filtering(
        self, 
        query: str,
        *,
        using_vector = 'semantic',
        rating_lower_bound: float = 7.8,
        popularity_lower_bound: float = 300
        
    ):
        query_embedding = self.encoders['semantic'].encode(query).tolist()

        filter_conditions = models.Filter(
            must=[
                models.FieldCondition(key='popularity', range=models.Range(gte=popularity_lower_bound, lte=float('inf'))),
                models.FieldCondition(key='vote_average', range=models.Range(gte=rating_lower_bound, lte=float('inf')))
            ]
        )

        # allow retrieval without indexing
        self.client.update_collection(
            collection_name=self.collection_name,
            strict_mode_config=models.StrictModeConfig(unindexed_filtering_retrieve=True)
        )

        # warmup 
        self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            using=using_vector,
            limit=1
        )

        times = []

        print("===Start search with filtering===")

        for i in range(25):
            start_time = time.time()
            _ = self.client.query_points(
                collection_name=self.collection_name,
                using=using_vector,
                query_filter=filter_conditions,
                limit=10,
                with_payload=False,
            )

            times.append((time.time() - start_time) * 1000)
            print(f"Time spend on query {times[-1]} for the cycle {i}/25")
        
        time_without_index = np.mean(times)

        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name='popularity',
            field_schema=models.PayloadSchemaType.INTEGER,
            wait=True
        )

        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name='vote_average',
            field_schema=models.PayloadSchemaType.INTEGER,
            wait=True
        )

        base_ef = self.client.get_collection(self.collection_name).config.hnsw_config.ef_construct

        new_ef_contruct = base_ef + 1

        self.client.update_collection(
            collection_name=self.collection_name,
            hnsw_config=models.HnswConfigDiff(ef_construct=new_ef_contruct),
            strict_mode_config=models.StrictModeConfig(
                unindexed_filtering_retrieve=False
            ) # turn off scanning and use indexing instead
        )

        self.wait_for_indexing(timeout=240)

        # warmup 
        self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            using=using_vector,
            limit=1
        )

        times_indexed = []

        for i in range(25):
            start_time = time.time()
            _ = self.client.query_points(
                collection_name=self.collection_name,
                using=using_vector,
                query_filter=filter_conditions,
                limit=10,
                with_payload=False,
            )

            times_indexed.append((time.time() - start_time) * 1000)
            print(f"Time spend on indexed query {times[-1]} for the cycle {i}/25")
        
        time_with_index = np.mean(times_indexed)


        return {
            "without_index": time_without_index,
            "with_index": time_with_index,
            "speedup": time_without_index / time_with_index
        }





        
    


