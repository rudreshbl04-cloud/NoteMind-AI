"""Gemini API client interactions for NoteMind AI.

Handles client initialization, resilient key validation, text embeddings (with batching & rate-limit handling),
and low-latency note-grounded answer generation with streaming and model caching.
"""

import time
from typing import Any, Dict, Generator, List, Optional, Tuple
from google import genai
from google.genai import types
from google.genai.errors import APIError, ClientError

# Global caches for working models discovered for the active session
WORKING_CHAT_MODEL: Optional[str] = None
WORKING_EMBEDDING_MODEL: Optional[str] = None
_CACHED_AVAILABLE_MODELS: Optional[Tuple[List[str], List[str]]] = None


def set_working_models(chat_model: Optional[str] = None, embedding_model: Optional[str] = None) -> None:
    """Set the globally active models discovered for the current session."""
    global WORKING_CHAT_MODEL, WORKING_EMBEDDING_MODEL
    if chat_model:
        WORKING_CHAT_MODEL = chat_model
    if embedding_model:
        WORKING_EMBEDDING_MODEL = embedding_model


def get_working_models() -> Tuple[str, str]:
    """Retrieve currently active chat and embedding model names."""
    global WORKING_CHAT_MODEL, WORKING_EMBEDDING_MODEL
    return (
        WORKING_CHAT_MODEL or "gemini-1.5-flash",
        WORKING_EMBEDDING_MODEL or "gemini-embedding-001",
    )


def clean_api_key(api_key: str) -> str:
    """Sanitize user-entered API key by stripping quotes, whitespace, and variable prefixes."""
    if not api_key:
        return ""
    key = api_key.strip()
    for prefix in [
        "GEMINI_API_KEY=",
        "gemini_api_key=",
        "GOOGLE_API_KEY=",
        "google_api_key=",
        "API_KEY=",
        "api_key=",
        "Bearer ",
    ]:
        if key.startswith(prefix):
            key = key[len(prefix):].strip()
    return key.strip("\"' \t\r\n")


def create_gemini_client(api_key: str) -> Optional[genai.Client]:
    """Create a Google GenAI Client using the user's API key.

    Never logs or persists the key.
    """
    sanitized = clean_api_key(api_key)
    if not sanitized:
        return None
    return genai.Client(api_key=sanitized)


def get_available_models(client: genai.Client, force_refresh: bool = False) -> Tuple[List[str], List[str]]:
    """Query Google GenAI ModelService.ListModels to find available models for this API key.

    Uses in-memory caching so repeated questions do not incur model-listing network latency.

    Returns:
        (chat_models, embedding_models): Sorted lists of available model identifiers.
    """
    global _CACHED_AVAILABLE_MODELS
    if _CACHED_AVAILABLE_MODELS is not None and not force_refresh:
        return _CACHED_AVAILABLE_MODELS

    chat_models: List[str] = []
    embed_models: List[str] = []

    model_pager = client.models.list(config={"page_size": 100})
    for m in model_pager:
        raw_name = getattr(m, "name", "") or ""
        clean_name = raw_name.replace("models/", "")
        actions = getattr(m, "supported_actions", []) or []

        if "generateContent" in actions:
            chat_models.append(clean_name)
        if "embedContent" in actions:
            embed_models.append(clean_name)

    # Sort chat models: prioritize fast flash models, then pro, then others
    priority_order = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-1.5-flash-latest",
        "gemini-1.5-flash-8b",
        "gemini-2.0-flash-lite",
        "gemini-1.5-pro",
        "gemini-1.5-pro-latest",
        "gemini-pro",
    ]

    sorted_chat = [m for m in priority_order if m in chat_models]
    sorted_chat += [m for m in chat_models if m not in sorted_chat]

    # Sort embedding models
    embed_priority = [
        "gemini-embedding-001",
        "text-embedding-005",
        "text-embedding-004",
        "embedding-001",
        "gemini-embedding-2",
    ]
    sorted_embed = [m for m in embed_priority if m in embed_models]
    sorted_embed += [m for m in embed_models if m not in sorted_embed]

    _CACHED_AVAILABLE_MODELS = (sorted_chat, sorted_embed)
    return _CACHED_AVAILABLE_MODELS


def validate_api_key(api_key: str) -> Tuple[bool, str]:
    """Perform a resilient check to verify the Gemini API key and discover available models.

    Probes candidate chat and embedding models with lightweight requests to lock in
    verified working models, eliminating trial-and-error latency during chat.

    Returns:
        (is_valid, message): A boolean flag and a beginner-friendly status message.
    """
    clean_key = clean_api_key(api_key)
    if not clean_key:
        return False, "Gemini API key is required. Please enter your key."

    try:
        client = create_gemini_client(clean_key)
        if not client:
            return False, "Gemini API key is required. Please enter your key."

        global WORKING_CHAT_MODEL, WORKING_EMBEDDING_MODEL
        chat_models, embed_models = get_available_models(client, force_refresh=True)

        # 1. Probe candidate chat models to guarantee instant response with no 404s
        chat_candidates: List[str] = []
        for preferred in [
            "gemini-2.0-flash",
            "gemini-1.5-flash",
            "gemini-1.5-flash-latest",
            "gemini-1.5-flash-8b",
            "gemini-2.0-flash-lite",
            "gemini-2.5-flash",
        ]:
            if preferred in chat_models and preferred not in chat_candidates:
                chat_candidates.append(preferred)
        for m in chat_models:
            if m not in chat_candidates:
                chat_candidates.append(m)
        if not chat_candidates:
            chat_candidates = ["gemini-2.0-flash", "gemini-1.5-flash"]

        working_chat = None
        for cand in chat_candidates:
            variants = [cand]
            if not cand.startswith("models/"):
                variants.append(f"models/{cand}")
            for variant in variants:
                try:
                    client.models.generate_content(
                        model=variant,
                        contents="hi",
                        config=types.GenerateContentConfig(max_output_tokens=1),
                    )
                    working_chat = variant
                    break
                except Exception:
                    continue
            if working_chat:
                break

        WORKING_CHAT_MODEL = working_chat or (chat_models[0] if chat_models else "gemini-1.5-flash")

        # 2. Probe candidate embedding models to guarantee embeddings succeed instantly
        embed_candidates: List[str] = []
        for preferred in ["gemini-embedding-001", "text-embedding-004", "text-embedding-005"]:
            if preferred in embed_models and preferred not in embed_candidates:
                embed_candidates.append(preferred)
        for m in embed_models:
            if m not in embed_candidates:
                embed_candidates.append(m)
        for fb in ["gemini-embedding-001", "text-embedding-004", "text-embedding-005"]:
            if fb not in embed_candidates:
                embed_candidates.append(fb)

        working_emb = None
        for cand in embed_candidates:
            variants = [cand]
            if not cand.startswith("models/"):
                variants.append(f"models/{cand}")
            for variant in variants:
                try:
                    res = client.models.embed_content(
                        model=variant,
                        contents="hi",
                    )
                    if getattr(res, "embeddings", None):
                        working_emb = variant
                        break
                except Exception:
                    continue
            if working_emb:
                break

        WORKING_EMBEDDING_MODEL = working_emb or (embed_models[0] if embed_models else "gemini-embedding-001")

        display_name = WORKING_CHAT_MODEL.replace("models/", "")
        return True, f"Connected successfully! (Using {display_name})"

    except ClientError as e:
        error_msg = str(e).lower()
        if any(term in error_msg for term in ["api_key_invalid", "api key not valid", "unauthenticated", "invalid api key", "key not found"]):
            return False, "Invalid Gemini API key. Please check the key in Google AI Studio and try again."
        if "permission_denied" in error_msg or "403" in error_msg:
            return False, "Access denied for this key. Please make sure the Generative Language API is enabled in your Google Cloud / AI Studio project."
        if "quota" in error_msg or "resource_exhausted" in error_msg or "429" in error_msg:
            return False, "Gemini API rate limit or quota exceeded. Please wait a moment and try again."
        return False, f"Authentication error: {e.message if hasattr(e, 'message') and e.message else 'Invalid key or access restricted.'}"

    except APIError as e:
        error_msg = str(e).lower()
        if "quota" in error_msg or "resource_exhausted" in error_msg or "429" in error_msg:
            return False, "Gemini API rate limit or quota exceeded. Please try again later."
        if any(term in error_msg for term in ["api_key_invalid", "api key not valid", "unauthenticated"]):
            return False, "Invalid Gemini API key. Please check the key in Google AI Studio and try again."
        return False, "Google Gemini API error. Please check your project settings in Google AI Studio."

    except Exception as e:
        error_name = type(e).__name__.lower()
        error_str = str(e).lower()
        if any(term in error_name or term in error_str for term in ["connect", "timeout", "network", "dns", "unreachable"]):
            return False, "Network error. Please check your internet connection and try again."
        return False, "Unable to verify API key. Please check your internet connection and try again."


def get_text_embedding(text: str, client: genai.Client) -> List[float]:
    """Generate an embedding for a single text (e.g. search query)."""
    embeddings = get_embeddings_for_chunks([text], client)
    if embeddings and len(embeddings) > 0:
        return embeddings[0]
    raise RuntimeError("Failed to generate embedding for query.")


def get_embeddings_for_chunks(
    texts: List[str],
    client: genai.Client,
    batch_size: int = 15,
) -> List[List[float]]:
    """Generate vector embeddings for a list of text chunks with batching and rate-limit retries.

    Args:
        texts: List of strings to embed.
        client: Active Google GenAI client.
        batch_size: Number of texts per API request to respect free tier RPM limits.

    Returns:
        List of embedding vectors corresponding to the input texts.
    """
    if not texts:
        return []

    global WORKING_EMBEDDING_MODEL

    # Retrieve cached embedding models
    _, discovered_embeds = get_available_models(client)
    models_to_try: List[str] = []
    if WORKING_EMBEDDING_MODEL:
        models_to_try.append(WORKING_EMBEDDING_MODEL)
    for m in discovered_embeds:
        if m not in models_to_try:
            models_to_try.append(m)
    for fallback in ["gemini-embedding-001", "text-embedding-004", "text-embedding-005", "embedding-001"]:
        if fallback not in models_to_try:
            models_to_try.append(fallback)

    all_embeddings: List[List[float]] = []

    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch = [t.strip() for t in texts[i : i + batch_size] if t.strip()]
        if not batch:
            continue

        batch_success = False
        last_error = None

        for model_name in models_to_try:
            model_variants = [model_name, f"models/{model_name}"]
            for mod_var in model_variants:
                for attempt in range(3):
                    try:
                        response = client.models.embed_content(
                            model=mod_var,
                            contents=batch if len(batch) > 1 else batch[0],
                        )

                        if response.embeddings:
                            if len(response.embeddings) == len(batch):
                                for emb in response.embeddings:
                                    all_embeddings.append(emb.values)
                                WORKING_EMBEDDING_MODEL = model_name
                                batch_success = True
                                break
                            elif len(batch) == 1 and len(response.embeddings) >= 1:
                                all_embeddings.append(response.embeddings[0].values)
                                WORKING_EMBEDDING_MODEL = model_name
                                batch_success = True
                                break

                    except Exception as e:
                        last_error = e
                        err_str = str(e).lower()
                        if "429" in err_str or "quota" in err_str or "resource_exhausted" in err_str:
                            time.sleep(2.0 * (attempt + 1))
                            continue
                        elif "404" in err_str or "not found" in err_str:
                            break
                        else:
                            break

                if batch_success:
                    break
            if batch_success:
                break

        # Fallback to item-by-item if batch call wasn't supported
        if not batch_success:
            for single_text in batch:
                item_success = False
                for model_name in models_to_try:
                    for mod_var in [model_name, f"models/{model_name}"]:
                        for attempt in range(3):
                            try:
                                response = client.models.embed_content(
                                    model=mod_var,
                                    contents=single_text,
                                )
                                if response.embeddings and len(response.embeddings) > 0:
                                    all_embeddings.append(response.embeddings[0].values)
                                    WORKING_EMBEDDING_MODEL = model_name
                                    item_success = True
                                    time.sleep(0.1)
                                    break
                            except Exception as e:
                                last_error = e
                                if "429" in str(e) or "quota" in str(e).lower():
                                    time.sleep(2.0 * (attempt + 1))
                                    continue
                                break
                        if item_success:
                            break
                    if item_success:
                        break

                if not item_success:
                    err_detail = str(last_error) if last_error else "Unknown embedding error"
                    raise RuntimeError(f"Embedding failed: {err_detail}")

    return all_embeddings


def generate_answer(prompt: str, client: genai.Client) -> str:
    """Generate an answer using the Gemini model with low temperature for strict grounding.

    Uses cached working model for ultra-low latency.
    """
    global WORKING_CHAT_MODEL

    discovered_chat, _ = get_available_models(client)

    models_to_try: List[str] = []
    if WORKING_CHAT_MODEL:
        models_to_try.append(WORKING_CHAT_MODEL)

    for m in discovered_chat:
        if m not in models_to_try:
            models_to_try.append(m)

    for fallback in [
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-1.5-flash-latest",
        "gemini-1.5-flash-8b",
        "gemini-2.0-flash-lite",
        "gemini-2.5-flash",
        "gemini-1.5-pro",
        "gemini-1.5-pro-latest",
        "gemini-pro",
    ]:
        if fallback not in models_to_try:
            models_to_try.append(fallback)

    last_error = None

    for model_name in models_to_try:
        variants = [model_name]
        if not model_name.startswith("models/"):
            variants.append(f"models/{model_name}")

        for variant in variants:
            try:
                response = client.models.generate_content(
                    model=variant,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        max_output_tokens=1024,
                    ),
                )
                WORKING_CHAT_MODEL = model_name
                return (response.text or "").strip()
            except ClientError as e:
                last_error = e
                err_str = str(e).lower()
                if "404" in err_str or "not found" in err_str or "not supported" in err_str:
                    continue
                if "429" in err_str or "quota" in err_str:
                    continue
                continue
            except Exception as e:
                last_error = e
                continue

    error_str = str(last_error).lower() if last_error else ""
    if "quota" in error_str or "resource_exhausted" in error_str:
        raise RuntimeError("Gemini API quota exceeded. Please wait a moment and try again.") from None
    if "connect" in error_str or "network" in error_str:
        raise ConnectionError("Network error. Please check your internet connection.") from None
    raise RuntimeError(f"Failed to generate response from Gemini API: {str(last_error)}") from None


def generate_answer_stream(prompt: str, client: genai.Client) -> Generator[str, None, None]:
    """Stream answer tokens in real-time to eliminate perceived waiting latency.

    Yields:
        Incremental text tokens as Gemini generates them.
    """
    global WORKING_CHAT_MODEL

    discovered_chat, _ = get_available_models(client)

    models_to_try: List[str] = []
    if WORKING_CHAT_MODEL:
        models_to_try.append(WORKING_CHAT_MODEL)

    for m in discovered_chat:
        if m not in models_to_try:
            models_to_try.append(m)

    for fallback in [
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-1.5-flash-latest",
        "gemini-1.5-flash-8b",
        "gemini-2.0-flash-lite",
        "gemini-2.5-flash",
        "gemini-1.5-pro",
        "gemini-1.5-pro-latest",
        "gemini-pro",
    ]:
        if fallback not in models_to_try:
            models_to_try.append(fallback)

    last_error = None

    for model_name in models_to_try:
        variants = [model_name]
        if not model_name.startswith("models/"):
            variants.append(f"models/{model_name}")

        for variant in variants:
            try:
                response = client.models.generate_content_stream(
                    model=variant,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        max_output_tokens=1024,
                    ),
                )
                yielded = False
                for chunk in response:
                    text_part = getattr(chunk, "text", "") or ""
                    if text_part:
                        yielded = True
                        WORKING_CHAT_MODEL = model_name
                        yield text_part
                if yielded:
                    return
            except ClientError as e:
                last_error = e
                err_str = str(e).lower()
                if "404" in err_str or "not found" in err_str or "not supported" in err_str:
                    continue
                continue
            except Exception as e:
                last_error = e
                continue

    # Fallback to non-streaming if stream encountered issues
    yield generate_answer(prompt, client)
