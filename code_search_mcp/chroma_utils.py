"""Helpers for working around ChromaDB local (rusqlite) backend limits."""
from typing import Any, Dict, List, Optional

# ChromaDB's local rusqlite backend binds one SQL variable per returned row when
# materialising a get() result. A single get() that returns more than SQLite's
# SQLITE_MAX_VARIABLE_NUMBER (32766) rows therefore fails with
# "Error executing plan: ... (code: 1) too many SQL variables".
# Read large collections in pages comfortably below that limit and stitch the
# pages back together. count() and query() are unaffected — only unbounded get().
CHROMA_GET_PAGE_SIZE = 10000


def get_all_records(
    collection,
    include: Optional[List[str]] = None,
    where: Optional[Dict[str, Any]] = None,
    page_size: int = CHROMA_GET_PAGE_SIZE,
) -> Dict[str, Any]:
    """Fetch every matching record from a ChromaDB collection, paginated.

    Equivalent to ``collection.get(include=..., where=...)`` but safe on
    collections larger than SQLite's variable limit. Returns a dict shaped like
    ``Collection.get()``: always ``{'ids': [...]}`` plus one list per entry in
    ``include`` (e.g. ``'metadatas'``, ``'documents'``, ``'embeddings'``), in the
    same order across all keys.

    Args:
        collection: A ChromaDB collection.
        include: Fields to return besides ids (e.g. ['metadatas', 'documents']).
            None or [] returns ids only.
        where: Optional metadata filter, forwarded to get().
        page_size: Rows per page; must stay below 32766.
    """
    include = list(include or [])
    merged: Dict[str, Any] = {'ids': []}
    for key in include:
        merged[key] = []

    offset = 0
    while True:
        kwargs: Dict[str, Any] = {'limit': page_size, 'offset': offset, 'include': include}
        if where is not None:
            kwargs['where'] = where
        page = collection.get(**kwargs)

        ids = page.get('ids') or []
        if not ids:
            break

        merged['ids'].extend(ids)
        for key in include:
            values = page.get(key)
            if values is not None:
                merged[key].extend(values)

        offset += len(ids)
        if len(ids) < page_size:
            break

    return merged


def delete_by_ids(collection, ids: List[str], page_size: int = CHROMA_GET_PAGE_SIZE) -> None:
    """Delete records by id in pages.

    Defensive: unbounded get() overflows SQLite's variable limit ("too many SQL
    variables"), and delete(ids=[...]) is the same WHERE id IN (...) shape. The
    local backend (chromadb 1.5.9) does handle large delete lists internally, but
    we page anyway — it's a no-op for small lists and guards the same error class
    across backend versions.
    """
    if not ids:
        return
    for start in range(0, len(ids), page_size):
        collection.delete(ids=ids[start:start + page_size])
