import json
from typing import Any
from urllib.parse import urlencode


class URLBuilder:
    def __init__(self, base_url: str) -> None:
        self._base_url = base_url
        self._params: dict[str, Any] = {}

    def add_param(self, key: str, value: Any) -> 'URLBuilder':
        """Add a single parameter"""
        self._params[key] = value
        return self
    
    def remove_param(self, key: str, value: Any) -> 'URLBuilder':
        """Remove a single parameter"""
        self._params.pop(key, None)
        return self
    
    def build(self) -> str:
        """Build the final URL"""
        if not self._params:
            return self._base_url

        encoded_params = {}
        for key, value in self._params.items():
            if isinstance(value, bool):
                encoded_params[key] = str(value).lower()
            elif isinstance(value, list):
                encoded_params[key] = [
                    str(v).lower() if isinstance(v, bool) else str(v) 
                    for v in value
                ]
            elif isinstance(value, dict):
                encoded_params[key] = json.dumps(value, ensure_ascii=False)
            elif value is not None:
                encoded_params[key] = str(value)
            
        query_str = urlencode(encoded_params, doseq=True)
        separator = '&' if '?' in self._base_url else '?'
        return f'{self._base_url}{separator}{query_str}'
