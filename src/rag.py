"""Retrieval-Augmented Generation (RAG) pipeline for NoteMind AI.

Coordinates question embedding, vector similarity retrieval, distance filtering,
prompt construction with strict grounding instructions, and source citation extraction.
"""

from typing import Any, Dict, List, Optional, Tuple
from google import genai

from src.gemini_client import get_text_embedding, generate_answer
from src.vector_store import search_similar_chunks, get_total_chunk_count

# Configurable constants
NOT_FOUND_MESSAGE = "Not found in your notes"
MAX_RELEVANCE_DISTANCE = 1.35  # Cosine distance cutoff (smaller = closer, > 1.35 considered irrelevant)
DEFAULT_RETRIEVAL_K = 5

# Strict grounding prompt template
RAG_PROMPT_TEMPLATE = """You are NoteMind AI, an accurate study assistant answering questions based strictly on the student's uploaded notes.

Instructions:
1. Base your answer ONLY on the provided Note Context excerpts below.
2. Provide a clear, thorough, and helpful answer explaining the relevant concepts found in the excerpts.
3. If the Note Context truly does not mention or contain information about the topic asked, respond with: "Not found in your notes."
4. Do not invent facts or use outside knowledge not supported by the context.

=== NOTE CONTEXT ===
{context_text}
===================

Question: {question}

Answer:"""


def format_context_and_sources(chunks: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """Format retrieved chunks into a prompt context string and extract unique citations."""
    context_blocks = []
    unique_sources = []
    seen = set()

    for idx, chunk in enumerate(chunks, start=1):
        source = chunk["source"]
        page = chunk["page"]
        text = chunk["text"]

        context_blocks.append(f"[Excerpt {idx} - From '{source}', Page {page}]:\n{text}")

        source_key = (source, page)
        if source_key not in seen:
            seen.add(source_key)
            unique_sources.append({
                "source": source,
                "page": page,
                "display": f"{source} — Page {page}",
            })

    return "\n\n".join(context_blocks), unique_sources


def prepare_rag_context(
    question: str,
    gemini_client: genai.Client,
    top_k: int = DEFAULT_RETRIEVAL_K,
    distance_threshold: float = MAX_RELEVANCE_DISTANCE,
) -> Tuple[Optional[str], List[Dict[str, Any]], Optional[str]]:
    """Retrieve relevant chunks and construct the grounded prompt.

    Returns:
        (prompt, sources, immediate_message):
        - If immediate_message is provided, no Gemini call is needed.
        - If prompt is provided, call Gemini to generate or stream the grounded answer.
    """
    clean_question = question.strip()
    if not clean_question:
        return None, [], "Please ask a question about your uploaded notes."

    if get_total_chunk_count() == 0:
        return None, [], "No notes have been uploaded yet. Please upload your PDF notes first."

    try:
        query_embedding = get_text_embedding(clean_question, gemini_client)
    except Exception as e:
        return None, [], f"Error searching notes: {str(e)}"

    retrieved_chunks = search_similar_chunks(query_embedding, top_k=top_k)
    if not retrieved_chunks:
        return None, [], NOT_FOUND_MESSAGE

    relevant_chunks = [
        chunk for chunk in retrieved_chunks
        if chunk.get("distance", 1.0) <= distance_threshold
    ]

    if not relevant_chunks:
        return None, [], NOT_FOUND_MESSAGE

    context_text, sources = format_context_and_sources(relevant_chunks)
    prompt = RAG_PROMPT_TEMPLATE.format(context_text=context_text, question=clean_question)
    return prompt, sources, None


def answer_question(
    question: str,
    gemini_client: genai.Client,
    top_k: int = DEFAULT_RETRIEVAL_K,
    distance_threshold: float = MAX_RELEVANCE_DISTANCE,
) -> Dict[str, Any]:
    """Execute the full RAG pipeline for a user question (non-streaming).

    Args:
        question: User query string.
        gemini_client: Active Google GenAI client.
        top_k: Number of relevant chunks to retrieve.
        distance_threshold: Maximum allowed cosine distance for relevance.

    Returns:
        A dictionary with 'answer', 'sources', and 'found'.
    """
    prompt, sources, immediate_msg = prepare_rag_context(
        question=question,
        gemini_client=gemini_client,
        top_k=top_k,
        distance_threshold=distance_threshold,
    )

    if immediate_msg is not None:
        return {
            "answer": immediate_msg,
            "sources": [],
            "found": False,
        }

    try:
        raw_answer = generate_answer(prompt, gemini_client)
    except Exception as e:
        return {
            "answer": f"Error generating answer: {str(e)}",
            "sources": [],
            "found": False,
        }

    cleaned_answer = raw_answer.strip()
    if cleaned_answer.lower().startswith(NOT_FOUND_MESSAGE.lower()) or not cleaned_answer:
        return {
            "answer": NOT_FOUND_MESSAGE,
            "sources": [],
            "found": False,
        }

    return {
        "answer": cleaned_answer,
        "sources": sources,
        "found": True,
    }
