"""Query and retrieval logic."""

from .embedder import embed_query
from .store import SentryStore


def search_clips(query: str, n_results: int = 5) -> list[dict]:
    """Search indexed video clips with a natural language query.

    Args:
        query: Natural language search string.
        n_results: Number of results to return.

    Returns:
        List of result dicts sorted by relevance score (descending).
    """
    query_embedding = embed_query(query)
    store = SentryStore()
    return store.search(query_embedding, n_results=n_results)
