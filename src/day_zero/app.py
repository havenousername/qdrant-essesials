from qdrant_client import QdrantClient, models 
import os 

URL = os.getenv('QDRANT_URL')
API_KEY = os.getenv('QDRANT_API_KEY')

if URL is None or API_KEY is None:
    raise RuntimeError("Cannot run application w/o url and api_key of qdrant")

client = QdrantClient(
    url=URL,
    api_key=API_KEY
)

# 