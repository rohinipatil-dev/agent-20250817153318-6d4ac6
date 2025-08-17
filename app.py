import os
import io
import math
import time
import json
import base64
from typing import List, Dict, Any, Tuple, Optional

import streamlit as st
import numpy as np

from openai import OpenAI

# Initialize a global OpenAI client. Will be re-instantiated if user sets API key later.
client = OpenAI()


# ---------------------------
# Utility and NLP Functions
# ---------------------------

def ensure_openai_client(api_key: Optional[str]) -> None:
    """
    Ensures the global OpenAI client is initialized with the given api_key.
    If api_key is None, uses environment variable or any existing configuration.
    """
    global client
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    # Recreate the client to pick up the latest API key
    client = OpenAI()


def read_text_from_pdf(file: io.BytesIO) -> str:
    """
    Attempts to read text from a PDF using PyPDF2 if available.
    Returns extracted text or raises an Exception if PyPDF2 is unavailable.
    """
    try:
        import PyPDF2  # type: ignore
    except Exception as e:
        raise RuntimeError("PyPDF2 is not installed. Please upload .txt or .md files, or install PyPDF2.") from e

    try:
        reader = PyPDF2.PdfReader(file)
        pages_text = []
        for page in reader.pages:
            txt = page.extract_text() or ""
            pages_text.append(txt)
        return "\n".join(pages_text)
    except Exception as e:
        raise RuntimeError(f"Failed to parse PDF: {e}") from e


def load_uploaded_files(files: List[Any]) -> List[Dict[str, Any]]:
    """
    Loads uploaded files and returns a list of dicts with keys:
    - source: filename
    - text: extracted text
    """
    docs = []
    for f in files:
        name = f.name
        suffix = name.lower().split(".")[-1]
        try:
            if suffix in ["txt", "md"]:
                text = f.read().decode("utf-8", errors="ignore")
            elif suffix == "pdf":
                text = read_text_from_pdf(f)
            else:
                st.warning(f"Unsupported file type for {name}. Only txt, md, pdf are supported.")
                continue
        except Exception as e:
            st.error(f"Error reading {name}: {e}")
            continue

        text = text.strip()
        if text:
            docs.append({"source": name, "text": text})
    return docs


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    """
    Splits text into overlapping chunks by character count.
    """
    if chunk_size <= 0:
        chunk_size = 1200
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 5)

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = end - overlap
    return chunks


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Computes cosine similarity between rows of a and rows of b.
    Returns a matrix of shape (a.shape[0], b.shape[0]).
    """
    # Normalize to unit vectors
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return np.dot(a_norm, b_norm.T)


def batched_embeddings(texts: List[str], model: str, batch_size: int = 64, show_progress: bool = False) -> List[List[float]]:
    """
    Computes embeddings for a list of texts using OpenAI embeddings endpoint in batches.
    """
    vectors: List[List[float]] = []
    total = len(texts)
    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]
        if show_progress:
            st.write(f"Embedding batch {i // batch_size + 1}/{math.ceil(total / batch_size)}...")
        emb = client.embeddings.create(
            model=model,
            input=batch
        )
        for item in emb.data:
            vectors.append(item.embedding)
        # small delay to be gentle with rate limits
        time.sleep(0.02)
    return vectors


def format_citation(md: Dict[str, Any]) -> str:
    """
    Formats metadata into a concise citation string.
    """
    src = md.get("source", "Unknown")
    chunk_id = md.get("chunk_id", 0)
    title = md.get("title") or ""
    if title:
        return f"{title} | {src} | chunk {chunk_id}"
    return f"{src} | chunk {chunk_id}"


def build_context_string(retrieved: List[Dict[str, Any]]) -> str:
    """
    Builds a context string for the LLM from retrieved chunks.
    """
    lines = []
    for i, item in enumerate(retrieved, start=1):
        md = item["metadata"]
        citation = format_citation(md)
        text = item["text"].strip()
        lines.append(f"[CITATION {i}] {citation}\n{text}")
    return "\n\n".join(lines)


def call_llm_chat(model: str, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
    """
    Calls OpenAI chat completions endpoint and returns the assistant message content.
    """
    response = client.chat.completions.create(
        model=model,  # "gpt-4" or "gpt-3.5-turbo"
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content


# ---------------------------
# Vector Index Management
# ---------------------------

def initialize_index() -> None:
    """
    Initializes index storage in session state.
    """
    if "index" not in st.session_state:
        st.session_state.index = {
            "embeddings": None,      # numpy array (N, D)
            "chunks": [],            # list of texts
            "metadatas": [],         # list of dicts
            "embedding_model": None, # string
            "dimension": None,       # int
            "num_chunks": 0
        }


def clear_index() -> None:
    """
    Clears the in-memory index.
    """
    st.session_state.index = {
        "embeddings": None,
        "chunks": [],
        "metadatas": [],
        "embedding_model": st.session_state.get("embedding_model"),
        "dimension": None,
        "num_chunks": 0
    }


def add_documents_to_index(
    docs: List[Dict[str, Any]],
    chunk_size: int,
    overlap: int,
    embedding_model: str
) -> None:
    """
    Chunks uploaded documents, embeds them, and adds to the in-memory index.
    """
    if not docs:
        return

    # Prepare chunks and metadata
    new_chunks: List[str] = []
    new_metadatas: List[Dict[str, Any]] = []
    for d in docs:
        source = d["source"]
        text = d["text"]
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        for idx, ch in enumerate(chunks):
            md = {
                "source": source,
                "chunk_id": idx,
                # Optional: users can include "title" headers in their .txt to help citation
                "title": None
            }
            # Attempt to infer title from first line if it looks like a heading
            first_line = ch.strip().splitlines()[0] if ch.strip().splitlines() else ""
            if len(first_line) < 120 and any(h in first_line.lower() for h in ["chapter", "canto", "verse", "text", "purport", "introduction", "preface"]):
                md["title"] = first_line.strip()
            new_chunks.append(ch)
            new_metadatas.append(md)

    # Compute embeddings
    vectors = batched_embeddings(new_chunks, model=embedding_model, batch_size=64, show_progress=False)
    emb_np = np.array(vectors, dtype=np.float32)

    # Update index
    idx = st.session_state.index
    if idx["embeddings"] is None:
        idx["embeddings"] = emb_np
        idx["chunks"] = new_chunks
        idx["metadatas"] = new_metadatas
        idx["embedding_model"] = embedding_model
        idx["dimension"] = emb_np.shape[1] if emb_np.size > 0 else None
        idx["num_chunks"] = emb_np.shape[0]
    else:
        if idx["embedding_model"] != embedding_model:
            st.warning("Embedding model changed. Rebuilding index with the new model.")
            clear_index()
            add_documents_to_index(docs, chunk_size, overlap, embedding_model)
            return
        idx["embeddings"] = np.vstack([idx["embeddings"], emb_np]) if idx["embeddings"] is not None else emb_np
        idx["chunks"].extend(new_chunks)
        idx["metadatas"].extend(new_metadatas)
        idx["num_chunks"] = idx["embeddings"].shape[0]


def search_index(query: str, top_k: int, embedding_model: str) -> List[Dict[str, Any]]:
    """
    Searches the in-memory index for the most similar chunks to the query.
    Returns a list of dicts with keys: text, metadata, score
    """
    idx = st.session_state.index
    if idx["embeddings"] is None or idx["num_chunks"] == 0:
        return []

    # Embed query
    q_emb = client.embeddings.create(model=embedding_model, input=[query]).data[0].embedding
    q_vec = np.array([q_emb], dtype=np.float32)

    # Compute cosine similarities
    sims = cosine_similarity_matrix(q_vec, idx["embeddings"])[0]  # shape: (N,)
    top_indices = np.argsort(-sims)[:top_k]
    results = []
    for rank, i in enumerate(top_indices, start=1):
        results.append({
            "text": idx["chunks"][i],
            "metadata": idx["metadatas"][i],
            "score": float(sims[i])
        })
    return results


# ---------------------------
# Prompts
# ---------------------------

def vedic_system_prompt() -> str:
    return (
        "You are a research assistant specialized in Vedic scriptures published by the Bhaktivedanta Book Trust (BBT): "
        "Bhagavad-gita As It Is, Srimad-Bhagavatam, Sri Caitanya-caritamrita, Nectar of Devotion, Nectar of Instruction, "
        "Back to Godhead magazine, and the writings/lectures/letters of A.C. Bhaktivedanta Swami Prabhupada.\n\n"
        "Ground your answers ONLY in the provided Context excerpts. For every claim or quote, add bracketed citations "
        "like [CITATION 1], [CITATION 2] that correspond to the provided context items. If the information required is not "
        "found in the Context, reply: 'Reference not found in provided corpus.' Optionally, you may offer a brief general "
        "perspective clearly labeled 'General knowledge (non-verified)'. Do not fabricate citations or sources. Be concise, "
        "precise, and faithful to the texts."
    )


def build_user_prompt(question: str, context: str) -> str:
    return (
        f"Question:\n{question}\n\n"
        f"Context (use these excerpts as the only authoritative basis for your answer):\n"
        f"{context if context.strip() else '[No context provided]'}\n\n"
        "Instructions:\n"
        "- Answer the question using ONLY the Context above.\n"
        "- Include bracketed citations like [CITATION X] for each claim tied to an excerpt.\n"
        "- If the Context does not contain the necessary information, respond with:\n"
        "  'Reference not found in provided corpus.' and optionally add a short 'General knowledge (non-verified)'."
    )


# ---------------------------
# Streamlit App
# ---------------------------

def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""
    if "model" not in st.session_state:
        st.session_state.model = "gpt-4"
    if "embedding_model" not in st.session_state:
        st.session_state.embedding_model = "text-embedding-3-small"
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.2
    if "chunk_size" not in st.session_state:
        st.session_state.chunk_size = 1200
    if "overlap" not in st.session_state:
        st.session_state.overlap = 200
    if "top_k" not in st.session_state:
        st.session_state.top_k = 5


def sidebar():
    with st.sidebar:
        st.header("Settings")

        # API Key
        api_key_input = st.text_input("OpenAI API Key", type="password", help="Your key is not stored on any server.")
        apply_key = st.button("Apply API Key")
        if apply_key:
            st.session_state.api_key = api_key_input.strip()
            ensure_openai_client(st.session_state.api_key)
            st.success("API key applied.")

        # Model choices
        st.session_state.model = st.selectbox("Chat Model", options=["gpt-4", "gpt-3.5-turbo"], index=0)
        st.session_state.embedding_model = st.selectbox("Embedding Model", options=["text-embedding-3-small", "text-embedding-3-large"], index=0)

        # Generation params
        st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, st.session_state.temperature, 0.05)

        # Indexing params
        with st.expander("Indexing Parameters"):
            st.session_state.chunk_size = st.number_input("Chunk Size (characters)", min_value=400, max_value=4000, value=st.session_state.chunk_size, step=100)
            st.session_state.overlap = st.number_input("Chunk Overlap (characters)", min_value=0, max_value=1000, value=st.session_state.overlap, step=50)
            st.session_state.top_k = st.slider("Top K Context Chunks", 1, 12, st.session_state.top_k)

        # Index management
        if st.button("Clear Index"):
            clear_index()
            st.info("Index cleared.")
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.info("Chat cleared.")

        # Index status
        idx = st.session_state.index
        st.caption(f"Indexed Chunks: {idx['num_chunks']}")


def main():
    st.set_page_config(page_title="Vedic Scripture RAG (BBT)", page_icon="📚", layout="wide")
    st.title("📚 Vedic Scripture Assistant (BBT RAG)")

    st.write(
        "Ask questions and get answers grounded in uploaded BBT texts (e.g., Bhagavad-gita As It Is, Srimad-Bhagavatam, "
        "Back to Godhead, etc.). Upload your corpus below and the assistant will cite the exact excerpts it used."
    )

    init_session()
    initialize_index()
    sidebar()

    # Document upload
    st.subheader("Upload Vedic Texts (TXT/MD/PDF)")
    uploaded_files = st.file_uploader(
        "Upload BBT books, articles, or notes. For best results, use clean .txt or .md files.",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True
    )
    col_a, col_b = st.columns([1, 1])
    with col_a:
        index_btn = st.button("Index Documents")
    with col_b:
        show_context = st.checkbox("Show retrieved context for each answer", value=False)

    if index_btn and uploaded_files:
        docs = load_uploaded_files(uploaded_files)
        if not docs:
            st.warning("No readable documents were uploaded.")
        else:
            with st.spinner("Indexing documents (chunk + embed)..."):
                add_documents_to_index(
                    docs=docs,
                    chunk_size=st.session_state.chunk_size,
                    overlap=st.session_state.overlap,
                    embedding_model=st.session_state.embedding_model
                )
            st.success(f"Indexed {len(st.session_state.index['chunks'])} chunks total.")

    # Chat interface
    st.subheader("Ask a Question")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_query = st.chat_input("Type your question about the scriptures...")
    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)

        # Retrieve context
        retrieved = search_index(user_query, top_k=st.session_state.top_k, embedding_model=st.session_state.embedding_model)
        context_str = build_context_string(retrieved) if retrieved else ""

        if show_context:
            with st.expander("Retrieved Context"):
                st.text(context_str if context_str.strip() else "No context available (no documents indexed).")

        # Build prompts
        sys_prompt = vedic_system_prompt()
        user_prompt = build_user_prompt(user_query, context_str)

        # Call LLM
        try:
            answer = call_llm_chat(
                model=st.session_state.model,
                system_prompt=sys_prompt,
                user_prompt=user_prompt,
                temperature=st.session_state.temperature
            )
        except Exception as e:
            answer = f"Error from model: {e}"

        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.write(answer)


if __name__ == "__main__":
    main()