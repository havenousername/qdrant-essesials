"""
Movie models
"""

from pydantic import BaseModel, Field


class RawMovie(BaseModel):
    """
    Raw movie datatype from moview retrieval step
    """
    adult: bool
    backdrop_path: str | None
    id: int | str
    original_language: str
    original_title: str
    overview: str
    popularity: float
    poster_path: str | None
    release_date: str
    title: str
    video: bool
    vote_average: float
    vote_count: float
    plot: str | None = Field(default=None)
