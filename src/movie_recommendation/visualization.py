"""
Visualization utilities for movie search results
"""
import textwrap
from typing import Any


def display_movie_result(result_dict: dict[str, Any], show_chunk: bool = True):
    """
    Display a movie search result using print statements
    
    Args:
        result_dict: Dictionary containing 'point' and 'score' keys
        show_chunk: Whether to display the matching chunk
    """
    point = result_dict['point']
    movie_data = point.payload
    
    print(f"\n{'='*80}")
    print(f"🎬 {movie_data.get('title', 'Unknown')}")
    print(f"{'='*80}")
    
    # Basic info
    print(f"📅 Release Date: {movie_data.get('release_date', 'N/A')}")
    print(f"⭐ Rating: {movie_data.get('vote_average', 0.0)}/10 ({movie_data.get('vote_count', 0)} votes)")
    print(f"🌍 Language: {movie_data.get('original_language', 'N/A').upper()}")
    
    # Overview
    print(f"\n📝 Overview:")
    overview = movie_data.get('overview', 'No overview available')
    wrapped_overview = textwrap.fill(overview, width=78, initial_indent="   ", subsequent_indent="   ")
    print(wrapped_overview)
    
    # Matching chunk
    if show_chunk and 'chunk' in movie_data:
        print(f"\n🔍 Matching Chunk:")
        chunk = movie_data.get('chunk', '')
        wrapped_chunk = textwrap.fill(chunk, width=78, initial_indent="   ", subsequent_indent="   ")
        print(wrapped_chunk)
        print(f"   Strategy: {movie_data.get('chunk_strategy', 'N/A')} | Chunk #{movie_data.get('chunk_index', 0) + 1}")
    
    # Scores
    print(f"\n📊 Search Scores:")
    text_score = result_dict.get('text_score', point.score)
    image_score = result_dict.get('image_score', 0.0)
    combined_score = result_dict.get('score', point.score)
    
    print(f"   Text Relevance:  {text_score:.3f} {get_score_bar(text_score)}")
    if image_score > 0:
        print(f"   Image Relevance: {image_score:.3f} {get_score_bar(image_score)}")
    print(f"   Combined Score:  {combined_score:.3f} {get_score_bar(combined_score)}")
    
    # Additional metadata
    if movie_data.get('poster_url'):
        print(f"\n🖼️  Poster: {movie_data['poster_url']}")


def get_score_bar(score: float, width: int = 20) -> str:
    """
    Create a text-based progress bar for score visualization
    
    Args:
        score: Score value between 0 and 1
        width: Width of the bar in characters
        
    Returns:
        Text representation of the score bar
    """
    filled = int(score * width)
    empty = width - filled
    
    if score > 0.7:
        color = "🟩"  # Green
    elif score > 0.5:
        color = "🟨"  # Yellow
    else:
        color = "🟥"  # Red
    
    bar = f"[{'█' * filled}{'░' * empty}] {color}"
    return bar


def display_search_results(results: list[dict[str, Any]], max_display: int = 5):
    """
    Display multiple search results
    
    Args:
        results: List of result dictionaries
        max_display: Maximum number of results to display
    """
    if not results:
        print("No results to display")
        return
    
    print(f"\nShowing top {min(len(results), max_display)} results:\n")
    
    for i, result in enumerate(results[:max_display]):
        display_movie_result(result)
        
        if i < len(results) - 1 and i < max_display - 1:
            print("\n" + "-" * 80 + "\n")


def display_search_summary(query: str, results: list[dict[str, Any]], strategy_scores: dict[str, float]):
    """
    Display a summary of the search operation
    
    Args:
        query: The search query
        results: List of result dictionaries
        strategy_scores: Average scores for each strategy
    """
    print(f"\n{'*'*80}")
    print(f"Search Query: '{query}'")
    print(f"Total Results: {len(results)}")
    print(f"{'*'*80}\n")
    
    if strategy_scores:
        print("Strategy Performance:")
        for strategy, score in sorted(strategy_scores.items(), key=lambda x: x[1], reverse=True):
            print(f"   {strategy.upper()}: {score:.3f} avg score")
        print()


def display_unique_movies(results: list[Any], limit: int = 10):
    """
    Display search results ensuring each movie appears only once
    
    Args:
        results: List of ScoredPoint objects
        limit: Maximum number of unique movies to display
    """
    seen_titles = set()
    unique_count = 0
    
    print(f"\nUnique Movies (top {limit}):\n")
    
    for point in results:
        if point.payload:
            title = point.payload.get('title')
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_count += 1
                
                print(f"{unique_count}. {title}")
                print(f"   Score: {point.score:.3f}")
                print(f"   Release: {point.payload.get('release_date', 'N/A')}")
                print(f"   Rating: {point.payload.get('vote_average', 0.0)}/10")
                overview = point.payload.get('overview', '')[:100]
                if overview:
                    print(f"   Overview: {overview}...")
                print()
                
                if unique_count >= limit:
                    break