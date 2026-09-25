"""NoteMind AI - Chat with your Notes (Version 1).

A beginner-friendly Streamlit web application where students connect their own
Gemini API key, upload PDF notes, and ask grounded questions with exact page citations.
"""

import streamlit as st
from src.gemini_client import (
    create_gemini_client,
    validate_api_key,
    clean_api_key,
    get_text_embedding,
    get_embeddings_for_chunks,
    generate_answer_stream,
)
from src.pdf_processor import process_pdf, MAX_FILE_SIZE_MB
from src.vector_store import (
    add_chunks_to_vector_store,
    get_indexed_files,
    get_total_chunk_count,
    reset_vector_store,
)
from src.rag import answer_question, prepare_rag_context, NOT_FOUND_MESSAGE

# Streamlit Page Configuration
st.set_page_config(
    page_title="NoteMind AI - Chat with your Notes",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "gemini_api_key" not in st.session_state:
    st.session_state["gemini_api_key"] = None

if "gemini_client" not in st.session_state:
    st.session_state["gemini_client"] = None

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "processed_files" not in st.session_state:
    st.session_state["processed_files"] = set()


# -----------------------------------------------------------------------------
# SCREEN 1: Gemini API Key Setup
# -----------------------------------------------------------------------------
def render_setup_screen():
    """Display the clean API Key Setup screen when no valid key is active."""
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("<h1 style='text-align: center;'>📚 NoteMind AI</h1>", unsafe_allow_html=True)
        st.markdown(
            "<p style='text-align: center; font-size: 1.15rem; color: #666;'>"
            "Chat with your own PDF notes using Gemini."
            "</p>",
            unsafe_allow_html=True,
        )
        st.markdown("---")

        st.subheader("Connect Your Gemini API Key")
        st.write(
            "To get started, enter your Google Gemini API key below. "
            "Your key is held strictly in session memory and is **never** saved to disk or databases."
        )

        st.info(
            "💡 Don't have an API key? Get one for free from Google AI Studio:\n\n"
            "[👉 Get Free Gemini API Key (aistudio.google.com)](https://aistudio.google.com/)"
        )

        api_key_input = st.text_input(
            "Gemini API Key",
            type="password",
            placeholder="Paste your Gemini API key here...",
            help="Your key is held only in memory for this session.",
        )

        connect_btn = st.button("Connect", type="primary", use_container_width=True)

        if connect_btn:
            clean_key = clean_api_key(api_key_input)
            if not clean_key:
                st.warning("Gemini API key is required. Please enter your key.")
            else:
                with st.spinner("Validating API key with Google Gemini..."):
                    is_valid, message = validate_api_key(clean_key)

                if is_valid:
                    st.session_state["gemini_api_key"] = clean_key
                    st.session_state["gemini_client"] = create_gemini_client(clean_key)
                    st.success("Connected successfully! Opening your workspace...")
                    st.rerun()
                elif "network" in message.lower():
                    st.warning(message)
                else:
                    st.error(message)


# -----------------------------------------------------------------------------
# SCREEN 2: Notes Chat
# -----------------------------------------------------------------------------
def render_chat_screen():
    """Display the main Notes Chat screen once authenticated."""
    # Header Bar
    head_col1, head_col2 = st.columns([3, 1])
    with head_col1:
        st.title("📚 NoteMind AI")
        st.caption("Chat with your notes — strictly grounded with page citations")
    with head_col2:
        st.markdown("<div style='text-align: right; padding-top: 15px;'>", unsafe_allow_html=True)
        st.markdown("🟢 **Gemini Connected**")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # ---------------------------------------------------------
    # Sidebar: File Management & Settings
    # ---------------------------------------------------------
    with st.sidebar:
        st.header("⚙️ Controls")

        # Disconnect Action
        if st.button("Disconnect API Key", use_container_width=True):
            st.session_state["gemini_api_key"] = None
            st.session_state["gemini_client"] = None
            st.session_state["chat_history"] = []
            st.session_state["processed_files"] = set()
            st.rerun()

        st.markdown("---")
        st.header("📄 Upload PDF Notes")
        st.caption(f"Max size: {MAX_FILE_SIZE_MB} MB per file. Text-based PDFs only.")

        uploaded_files = st.file_uploader(
            "Select one or more PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload lecture notes, textbooks, or study materials.",
        )

        # Indexing logic for newly uploaded files
        if uploaded_files:
            new_files_to_process = [
                f for f in uploaded_files
                if f.name not in st.session_state["processed_files"]
            ]

            if new_files_to_process:
                process_btn = st.button("Index Uploaded Notes", type="primary", use_container_width=True)
                if process_btn:
                    client = st.session_state["gemini_client"]
                    for uploaded_file in new_files_to_process:
                        file_bytes = uploaded_file.getvalue()
                        fname = uploaded_file.name

                        with st.spinner(f"Reading '{fname}'..."):
                            chunks, err = process_pdf(file_bytes, fname)

                        if err:
                            st.error(err)
                            continue

                        with st.spinner(f"Generating embeddings for {len(chunks)} chunks in '{fname}'..."):
                            try:
                                chunk_texts = [c["text"] for c in chunks]
                                embeddings = get_embeddings_for_chunks(chunk_texts, client)
                                add_chunks_to_vector_store(chunks, embeddings)
                                st.session_state["processed_files"].add(fname)
                                st.success(f"Indexed '{fname}' ({len(chunks)} chunks)")
                            except Exception as e:
                                st.error(f"Embedding failed for '{fname}': {str(e)}")

        # Display currently indexed files
        st.markdown("---")
        st.subheader("📚 Indexed Notes")
        indexed_files = get_indexed_files()
        total_chunks = get_total_chunk_count()

        if indexed_files:
            for file_name in indexed_files:
                st.write(f"• **{file_name}**")
            st.caption(f"Total chunks indexed: {total_chunks}")
        else:
            st.info("No notes currently indexed.")

        st.markdown("---")

        # Clear Chat Button
        if st.button("Clear Chat", use_container_width=True):
            st.session_state["chat_history"] = []
            st.rerun()

        # Remove Files Confirmation & Button
        with st.expander("🗑️ Remove Files"):
            st.caption("Remove all indexed notes from local vector storage.")
            confirm_remove = st.checkbox("Confirm removal of notes", value=False)
            if st.button("Remove all files", type="secondary", disabled=not confirm_remove, use_container_width=True):
                reset_vector_store()
                st.session_state["processed_files"] = set()
                st.success("All notes removed from vector storage.")
                st.rerun()

    # ---------------------------------------------------------
    # Main Chat Area
    # ---------------------------------------------------------
    # Display Chat History
    for message in st.session_state["chat_history"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                st.markdown("**Sources:**")
                for src in message["sources"]:
                    st.markdown(f"- `{src['display']}`")

    # Chat Input
    user_query = st.chat_input("Ask a question about your uploaded notes...")

    if user_query:
        query_text = user_query.strip()
        if not query_text:
            return

        # Render User Message immediately
        st.session_state["chat_history"].append({
            "role": "user",
            "content": query_text,
            "sources": [],
        })
        with st.chat_message("user"):
            st.markdown(query_text)

        # Check if any notes are indexed
        if get_total_chunk_count() == 0:
            notice = "Please upload and index your PDF notes first before asking questions."
            st.session_state["chat_history"].append({
                "role": "assistant",
                "content": notice,
                "sources": [],
            })
            with st.chat_message("assistant"):
                st.info(notice)
            return

        # Perform low-latency RAG retrieval and real-time streaming generation
        with st.chat_message("assistant"):
            prompt, sources, immediate_msg = prepare_rag_context(
                question=query_text,
                gemini_client=st.session_state["gemini_client"],
            )

            if immediate_msg is not None:
                answer_text = immediate_msg
                st.markdown(answer_text)
                sources = []
            else:
                try:
                    stream_gen = generate_answer_stream(
                        prompt=prompt,
                        client=st.session_state["gemini_client"],
                    )
                    answer_text = st.write_stream(stream_gen)

                    if NOT_FOUND_MESSAGE.lower() in answer_text.lower():
                        sources = []
                except Exception as e:
                    answer_text = f"Error generating answer: {str(e)}"
                    st.error(answer_text)
                    sources = []

            if sources:
                st.markdown("**Sources:**")
                for src in sources:
                    st.markdown(f"- `{src['display']}`")

            # Store in chat history
            st.session_state["chat_history"].append({
                "role": "assistant",
                "content": answer_text,
                "sources": sources,
            })


# -----------------------------------------------------------------------------
# Main Application Router
# -----------------------------------------------------------------------------
def main():
    """Route between Screen 1 (Setup) and Screen 2 (Chat)."""
    if st.session_state.get("gemini_api_key"):
        render_chat_screen()
    else:
        render_setup_screen()


if __name__ == "__main__":
    main()
