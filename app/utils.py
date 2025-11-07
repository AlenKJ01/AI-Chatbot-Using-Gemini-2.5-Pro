import os
import re
import uuid
from typing import List, Tuple
from pathlib import Path
from numpy.linalg import norm
import numpy as np
import pdfplumber
import docx

from transformers import pipeline
from sentence_transformers import SentenceTransformer

# Optional: Google Gemini client
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_MODEL_NAME = os.getenv('GEMINI_MODEL_NAME', 'gemini-2.5-pro')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Directories
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / 'models'
MODEL_DIR.mkdir(exist_ok=True)

# --- Document extraction (no fitz, stable on Windows) ---

def load_document(file_path: str) -> str:
    """Extract text from PDF, DOCX, or TXT using pdfplumber/docx, and return cleaned string."""
    path = Path(file_path)
    text = ""
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        pages = []
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                content = page.extract_text() or ""
                pages.append(content)
        text = "\n".join(pages)

    elif suffix in (".docx", ".doc"):
        doc = docx.Document(str(path))
        paragraphs = [p.text for p in doc.paragraphs]
        text = "\n".join(paragraphs)

    elif suffix == ".txt":
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    else:
        raise ValueError("Unsupported file type: " + suffix)

    # Basic cleanup
    text = re.sub(r"\n{2,}", "\n\n", text).strip()
    return text


# --- Summarization (chunked) ---

SUMMARIZER = None
SUMMARIZER_MODEL = "facebook/bart-large-cnn"

def _get_summarizer():
    global SUMMARIZER
    if SUMMARIZER is None:
        SUMMARIZER = pipeline("summarization", model=SUMMARIZER_MODEL, device=-1)
    return SUMMARIZER


def chunk_text(text: str, max_tokens: int = 800) -> List[str]:
    """Split by paragraphs, maintaining approx chunk size."""
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    chunks = []
    current, current_len = [], 0
    for p in paragraphs:
        pl = len(p)
        if current_len + pl > max_tokens and current:
            chunks.append('\n'.join(current))
            current = [p]
            current_len = pl
        else:
            current.append(p)
            current_len += pl
    if current:
        chunks.append('\n'.join(current))
    return chunks


def summarize_document(document_text: str, max_chunk_chars: int = 1500) -> str:
    """Summarize text using BART in chunks."""
    summarizer = _get_summarizer()
    chunks = chunk_text(document_text, max_tokens=max_chunk_chars)

    chunk_summaries = []
    for c in chunks:
        try:
            out = summarizer(c, max_length=200, min_length=40, do_sample=False)
            chunk_summaries.append(out[0]['summary_text'].strip())
        except Exception:
            truncated = c[:1500]
            out = summarizer(truncated, max_length=200, min_length=40, do_sample=False)
            chunk_summaries.append(out[0]['summary_text'].strip())

    combined = "\n\n".join(chunk_summaries)
    if len(combined) > 1000:
        final = summarizer(combined, max_length=300, min_length=80, do_sample=False)[0]['summary_text']
    else:
        final = combined
    return final


# --- Embeddings & Pure-Python Semantic Search (FAISS-free) ---

EMBED_MODEL = None

def _get_embed_model():
    global EMBED_MODEL
    if EMBED_MODEL is None:
        EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    return EMBED_MODEL


def create_vectorstore(document_text: str, chunk_size: int = 800, overlap: int = 100) -> Tuple[str, dict]:
    """Split text into chunks, embed them, and store embeddings as numpy arrays."""
    text = document_text.strip()
    if not text:
        raise ValueError('Empty document')

    chunks = []
    start, L = 0, len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == L:
            break
        start = max(0, end - overlap)

    model = _get_embed_model()
    emb = model.encode(chunks, show_progress_bar=False)
    emb = np.array(emb).astype('float32')

    uid = uuid.uuid4().hex
    index_path = str(MODEL_DIR / f'vectorstore_{uid}.npy')
    np.save(index_path, emb)

    metadata = {'chunks': chunks, 'index_path': index_path}
    return index_path, metadata


def load_vectorstore(index_path: str):
    """Load numpy embeddings from file."""
    return np.load(index_path)


def semantic_search(index_array: np.ndarray, query: str, top_k: int = 4):
    """Perform cosine similarity search over numpy embeddings."""
    model = _get_embed_model()
    q_emb = model.encode([query]).astype('float32')[0]
    sims = index_array @ q_emb / (norm(index_array, axis=1) * norm(q_emb) + 1e-10)
    top_ids = np.argsort(-sims)[:top_k]
    top_scores = sims[top_ids]
    return top_ids, top_scores


# --- Gemini wrapper ---
def call_gemini(prompt: str,
                conversation_history: List[Tuple[str, str]] = None,
                max_output_tokens: int = 512) -> str:
    """Send prompt to Gemini 2.5 Pro using the latest SDK and handle token overflows."""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not configured. Set GEMINI_API_KEY in environment.")

    system_instruction = (
        "You are a helpful assistant. Answer the user's question using ONLY the provided CONTEXT below. "
        "If the answer is not in the context, say you don't know."
    )

    # build chat history
    history = [{"role": "user", "parts": system_instruction}]
    if conversation_history:
        for user_msg, assistant_msg in conversation_history:
            history.append({"role": "user", "parts": user_msg})
            history.append({"role": "model", "parts": assistant_msg})

    model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    chat = model.start_chat(history=history)

    # first attempt
    response = chat.send_message(
        prompt,
        generation_config={
            "max_output_tokens": max_output_tokens,
            "temperature": 0.0,
        },
    )

    # --- try to extract text safely ---
    text = None
    try:
        if hasattr(response, "text") and response.text:
            text = response.text
        elif getattr(response, "candidates", None):
            parts = response.candidates[0].content.parts
            text = "".join(p.text for p in parts if hasattr(p, "text") and p.text)
    except Exception:
        text = None

    # --- handle truncation / empty output ---
    finish_reason = None
    try:
        finish_reason = getattr(response, "result", {}).get("candidates", [{}])[0].get("finish_reason")
    except Exception:
        pass

    if not text or finish_reason == "MAX_TOKENS":
        # try again with fewer tokens
        try:
            retry = chat.send_message(
                prompt,
                generation_config={
                    "max_output_tokens": max_output_tokens // 2,
                    "temperature": 0.0,
                },
            )
            text = getattr(retry, "text", None) or str(retry)
        except Exception as e:
            text = f"[Gemini retry failed: {e}]"

    return (text or "[No content returned by Gemini]").strip()




# --- Conversational retrieval ---

def answer_query(index_path: str, metadata: dict, query: str, conversation_history: List[Tuple[str,str]] = None) -> Tuple[str, dict]:
    """Run semantic search over numpy embeddings, create prompt, call Gemini."""
    index = load_vectorstore(index_path)
    ids, distances = semantic_search(index, query, top_k=4)
    retrieved = [metadata['chunks'][i] for i in ids if i < len(metadata['chunks'])]

    context = "\n\n---\n\n".join(retrieved)
    prompt = (
        "CONTEXT:\n" + context +
        "\n\nQUESTION:\n" + query +
        "\n\nINSTRUCTIONS:\n"
        "Answer using ONLY the CONTEXT. If the answer is not present, say you don't know. Keep the answer concise."
    )

    answer = call_gemini(prompt, conversation_history=conversation_history)
    extra = {
        'retrieved_count': len(retrieved),
        'distances': distances.tolist() if hasattr(distances, 'tolist') else distances,
    }
    return answer, extra
