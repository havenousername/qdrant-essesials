# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Qdrant vector search system implementing a 16 personalities-based personality mapping system for analyzing person-position alignment in organizations.

## Development Setup

### Environment Variables
Required environment variables:
- `QDRANT_URL`: URL to Qdrant instance
- `QDRANT_API_KEY`: API key for Qdrant authentication

### Dependencies
Project uses Poetry for dependency management:
```bash
poetry install          # Install all dependencies
poetry install --with dev  # Include development dependencies (ipykernel)
poetry shell           # Activate virtual environment
```

### Running Notebooks
```bash
poetry run jupyter notebook  # Start Jupyter notebook server
```

## Architecture

### Vector Dimensions
The system uses 4-dimensional personality vectors representing:
1. **Abstract Thinking** (0-1): concrete to abstract thinking capability
2. **Emotional Awareness** (0-1): low to high emotional intelligence
3. **Risk Tolerance** (0-1): cautious to risk-taking behavior
4. **Structure Needness** (0-1): flexible to highly structured preference

### Key Components

- **app.py**: Basic Qdrant client setup with environment variable validation
- **challange.ipynb**: Main implementation notebook containing:
  - Collection creation with cosine distance metric
  - Personality data points with position assignments
  - Ideal position vectors for role matching
  - Outlier detection algorithm to identify person-position misalignments

### Collection Structure
- Collection name: `personal_personalities`
- Vector size: 4 dimensions
- Distance metric: Cosine
- Payload includes: name, position, test_last_updated

### Position Mapping
The system defines ideal personality vectors for positions like Software Developer, HR Manager, Creative Director, etc., enabling similarity searches to identify:
- Well-aligned employees
- Potential role mismatches
- Career transition opportunities

## Common Tasks

### Query for Personality Match
```python
client.query_points(
    collection_name,
    query=[0.8, 0.3, 0.5, 0.6],  # Example: Abstract thinker, low emotion, moderate risk, some structure
    limit=5
)
```

### Filter by Position
```python
client.query_points(
    collection_name,
    query=query_vector,
    query_filter=models.Filter(
        must=[models.FieldCondition(key="position", match=models.MatchValue(value="Data Scientist"))]
    )
)
```

### Detect Outliers
Compare each person's vector against ideal position vectors to identify misalignments with similarity scores > 0.7 for different positions.