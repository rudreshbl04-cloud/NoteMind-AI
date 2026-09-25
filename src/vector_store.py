"""Local vector database management using ChromaDB for NoteMind AI.

Handles local collection persistence, vector insertions with source metadata,
duplicate prevention, similarity search, and collection clearing.
"""

import os
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.config import Settings

# Configuration constants
CHROMA_PERSIST_DIR = os.path.join("data", "chroma")
COLLECTION_NAME = "notemind_notes"
DEFAULT_TOP_K = 4


def get_chroma_client(persist_directory: str = CHROMA_PERSIST_DIR) -> chromadb.ClientAPI:
    """Initialize and return a local persistent ChromaDB client."""
    os.makedirs(persist_directory, exist_ok=True)
    return chromadb.PersistentClient(
        path=persist_directory,
        settings=Settings(anonymized_telemetry=False),
    )


def get_notes_collection(client: Optional[chromadb.ClientAPI] = None) -> chromadb.Collection:
    """Get or create the stable collection for note chunks with cosine similarity."""
    if client is None:
        client = get_chroma_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks_to_vector_store(
    chunks: List[Dict[str, Any]],
    embeddings: List[List[float]],
    client: Optional[chromadb.ClientAPI] = None,
) -> int:
    """Insert text chunks and their embeddings into ChromaDB with source metadata.

    Prevents duplicate chunks if a file is re-indexed.

    Args:
        chunks: List of chunk dicts, each with 'text', 'source', 'page'.
        embeddings: List of embedding vectors matching the chunks.
        client: Optional active ChromaDB client.

    Returns:
        Number of chunks added.
    """
    if not chunks or not embeddings:
        return 0

    if len(chunks) != len(embeddings):
        raise ValueError("Number of chunks and embeddings must match.")

    collection = get_notes_collection(client)

    # Check for existing filenames to avoid duplicate entries for the same file
    filenames_to_add = set(chunk["source"] for chunk in chunks)
    for fname in filenames_to_add:
        # Delete existing entries for this file to ensure clean replacement
        delete_chunks_by_filename(fname, client=client)

    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        doc_id = f"{chunk['source']}_p{chunk['page']}_{index}"
        ids.append(doc_id)
        documents.append(chunk["text"])
        metadatas.append({
            "source": str(chunk["source"]),
            "page": int(chunk["page"]),
        })

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return len(documents)


def delete_chunks_by_filename(filename: str, client: Optional[chromadb.ClientAPI] = None) -> None:
    """Remove all chunks associated with a specific file from ChromaDB."""
    collection = get_notes_collection(client)
    try:
        collection.delete(where={"source": filename})
    except Exception:
        # If no documents match or collection is empty, ignore
        pass


def search_similar_chunks(
    query_embedding: List[float],
    top_k: int = DEFAULT_TOP_K,
    client: Optional[chromadb.ClientAPI] = None,
) -> List[Dict[str, Any]]:
    """Search for the most relevant note chunks given a query embedding.

    Args:
        query_embedding: Vector representation of the user question.
        top_k: Number of most similar chunks to retrieve.
        client: Optional active ChromaDB client.

    Returns:
        List of dicts containing 'text', 'source', 'page', and 'distance'.
    """
    collection = get_notes_collection(client)
    total_docs = collection.count()

    if total_docs == 0:
        return []

    effective_k = min(top_k, total_docs)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=effective_k,
        include=["documents", "metadatas", "distances"],
    )

    retrieved: List[Dict[str, Any]] = []
    if not results or not results.get("documents") or not results["documents"][0]:
        return []

    docs = results["documents"][0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(docs, metas, distances):
        retrieved.append({
            "text": doc,
            "source": meta.get("source", "Unknown"),
            "page": meta.get("page", 1),
            "distance": dist,
        })

    return retrieved


def reset_vector_store(client: Optional[chromadb.ClientAPI] = None) -> None:
    """Completely clear all indexed note chunks from ChromaDB."""
    if client is None:
        client = get_chroma_client()
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass
    # Re-create empty collection ready for new uploads
    get_notes_collection(client)


def get_indexed_files(client: Optional[chromadb.ClientAPI] = None) -> List[str]:
    """Return a unique list of filenames currently stored in ChromaDB."""
    collection = get_notes_collection(client)
    total = collection.count()
    if total == 0:
        return []

    data = collection.get(include=["metadatas"])
    if not data or not data.get("metadatas"):
        return []

    filenames = set()
    for meta in data["metadatas"]:
        if meta and "source" in meta:
            filenames.add(meta["source"])

    return sorted(list(filenames))


def get_total_chunk_count(client: Optional[chromadb.ClientAPI] = None) -> int:
    """Return total number of chunks currently indexed in ChromaDB."""
    collection = get_notes_collection(client)
    return collection.count()
