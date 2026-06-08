import os
from typing import Any, final

import requests
import wikipediaapi

from models.movie_models import RawMovie
from utils.urlbuilder import URLBuilder


@final
class MoviesDataset:
    _URL = 'https://api.themoviedb.org/3'
    def __init__(self, api_key_name: str) -> None:
        self.url = MoviesDataset._URL
        api_key = os.getenv(api_key_name) 

        if api_key is None:
            raise ValueError("Please provider correct name of the env for the api_key")

        self._api_key = api_key
        self._wikipedia = wikipediaapi.Wikipedia("MovieSearch/1.0", "en")


    def get_latest(self, *, page: int = 1, size: int = 20):
        print(f"Fetching latest movies - starting page: {page}, total size: {size}")
        url_builder = (URLBuilder(f'{self.url}/discover/movie')
            .add_param('include_adult', False)
            .add_param('include_video', False)
            .add_param('language', 'en-US')
            .add_param('sort_by', 'popularity.desc')
        )
        current_page = page
        headers = {
            'accept': 'application/json',
            'Authorization': f'Bearer {self._api_key}'
        }
        movies: list[RawMovie] = []
        while size > 0:
            url = (url_builder
                .add_param('page', current_page)
                .build()
            )
            print(f"Fetching page {current_page}...")
            response = requests.get(url, headers=headers)
            res_json: dict[str, Any] = response.json()
            raw_movies = res_json['results']
            print(f"Got {len(raw_movies)} movies from page {current_page}")
            for movie in raw_movies:
                movies.append(RawMovie(**movie))
            size -= 20
            current_page += 1

        print(f"Total movies fetched: {len(movies)}")
        return movies

    def get_enhanced_latest(self, *, page: int = 1, size: int = 20):
        movies = self.get_latest(page=page, size=size)
        print(f"\nEnhancing {len(movies)} movies with Wikipedia plots...")
        enhanced_count = 0
        for i, movie in enumerate(movies):
            print(f"Processing movie {i+1}/{len(movies)}: {movie.title}")
            try:
                wiki_page = self._wikipedia.page(movie.title)
                plot_section = wiki_page.section_by_title("Plot")
                if plot_section:
                    text = plot_section.text
                    movie.plot = text
                    enhanced_count += 1
                    print(f"  ✓ Found plot for {movie.title}")
                else:
                    print(f"  ✗ No plot section found for {movie.title}")
            except Exception as e:
                print(f"  ✗ Error fetching Wikipedia data for {movie.title}: {e}")
        print(f"\nEnhanced {enhanced_count}/{len(movies)} movies with plots")
        return movies
