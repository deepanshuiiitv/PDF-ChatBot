import os
import pickle
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_PATH = os.path.join(os.getcwd(), "model", "llama-2-7b-chat.ggmlv3.q4_0.bin")
VECTORSTORE_PATH = os.path.join(os.getcwd(), "vectorstore.pkl")

# in-memory stores
DOCUMENTS = []
EMBEDDINGS = None
NN = None
embed_model = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    import numpy as np
    from sklearn.neighbors import NearestNeighbors
except Exception:
    np = None
    NearestNeighbors = None


def get_embedder():
    global embed_model
    if embed_model is None:
        if SentenceTransformer is None:
            raise RuntimeError("Missing 'sentence-transformers'. Install with: pip install sentence-transformers scikit-learn")
        embed_model = SentenceTransformer(EMBEDDING_MODEL)
    return embed_model


def _persist():
    try:
        with open(VECTORSTORE_PATH, "wb") as f:
            pickle.dump({"documents": DOCUMENTS, "embeddings": EMBEDDINGS}, f)
    except Exception:
        pass


def load_vectorstore():
    global DOCUMENTS, EMBEDDINGS, NN
    if os.path.exists(VECTORSTORE_PATH):
        try:
            with open(VECTORSTORE_PATH, "rb") as f:
                data = pickle.load(f)
                DOCUMENTS = data.get("documents", [])
                EMBEDDINGS = data.get("embeddings", None)
                if EMBEDDINGS is not None and NearestNeighbors is not None:
                    NN = NearestNeighbors(n_neighbors=min(4, len(EMBEDDINGS)), metric="cosine")
                    NN.fit(EMBEDDINGS)
        except Exception:
            pass


def process_pdf(pdf_path):
    global DOCUMENTS, EMBEDDINGS, NN
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    chunks = splitter.split_documents(pages)

    texts = [c.page_content for c in chunks]

    embedder = get_embedder()
    embs = embedder.encode(texts, convert_to_numpy=True)

    # Try optional Pinecone upsert if configured, otherwise persist locally
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    PINECONE_INDEX = os.getenv('PINECONE_INDEX')
    used_pinecone = False

    if PINECONE_API_KEY and PINECONE_INDEX:
        try:
            from pinecone import Pinecone
            pc = Pinecone(api_key=PINECONE_API_KEY)
            idx = pc.Index(PINECONE_INDEX)
            vectors = []
            for i, emb in enumerate(embs):
                vals = emb.tolist() if hasattr(emb, 'tolist') else list(map(float, emb))
                vectors.append({'id': f"{os.path.basename(pdf_path)}_{i}", 'values': vals, 'metadata': {'text': texts[i]}})
            idx.upsert(vectors=vectors)
            used_pinecone = True
        except Exception:
            used_pinecone = False

    # Always keep a local copy so local search is available (helps when Pinecone is used or offline)
    if EMBEDDINGS is None:
        EMBEDDINGS = embs
        DOCUMENTS = texts
    else:
        EMBEDDINGS = np.vstack([EMBEDDINGS, embs])
        DOCUMENTS.extend(texts)

    if NearestNeighbors is not None:
        NN = NearestNeighbors(n_neighbors=min(4, len(DOCUMENTS)), metric="cosine")
        NN.fit(EMBEDDINGS)

    _persist()


# load on import if available
load_vectorstore()


def ask_question(query: str) -> str:
    # If there's no local vectorstore, only proceed if Pinecone is configured
    if EMBEDDINGS is None or len(DOCUMENTS) == 0:
        PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
        PINECONE_INDEX = os.getenv('PINECONE_INDEX')
        if not (PINECONE_API_KEY and PINECONE_INDEX):
            return "No PDF loaded yet."

    try:
        embedder = get_embedder()
    except Exception as e:
        # If embedder is required for both local search and Pinecone query, return helpful message
        return f"Cannot compute query embedding: {e}"

    q_emb = embedder.encode([query], convert_to_numpy=True)

    # If we have a local search index, use it
    if NN is not None:
        dists, idxs = NN.kneighbors(q_emb, n_neighbors=min(4, len(DOCUMENTS)))
        contexts = [DOCUMENTS[idx] for idx in idxs[0]]
        combined = "\n\n---\n\n".join(contexts)
    else:
        # Try Pinecone as a fallback if configured and return detailed diagnostics
        combined = None
        pinecone_diag = []
        try:
            PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
            PINECONE_INDEX = os.getenv('PINECONE_INDEX')
            if PINECONE_API_KEY and PINECONE_INDEX:
                try:
                    from pinecone import Pinecone
                    pc = Pinecone(api_key=PINECONE_API_KEY)
                    idx = pc.Index(PINECONE_INDEX)
                    vec = q_emb[0].tolist() if hasattr(q_emb[0], 'tolist') else list(map(float, q_emb[0]))
                    qres = idx.query(vector=vec, top_k=4, include_metadata=True)

                    matches = []
                    if hasattr(qres, 'matches'):
                        matches = qres.matches
                    elif isinstance(qres, dict) and 'matches' in qres:
                        matches = qres['matches']

                    contexts = []
                    for m in matches:
                        meta = getattr(m, 'metadata', None) or (m.get('metadata') if isinstance(m, dict) else {}) or {}
                        text = meta.get('text') or meta.get('content') or ''
                        if text:
                            contexts.append(text)

                    pinecone_diag.append(f"Pinecone returned {len(matches)} matches")
                    if contexts:
                        combined = "\n\n---\n\n".join(contexts)
                    else:
                        pinecone_diag.append("No usable text metadata found in matches")
                except Exception as e:
                    pinecone_diag.append(f"Pinecone query failed: {e}")
        except Exception as e:
            pinecone_diag.append(f"Pinecone fallback error: {e}")

        if combined is None:
            # Provide a helpful diagnostic instead of a generic 'No search index available.'
            diag = "; ".join(pinecone_diag) if pinecone_diag else "No local index and Pinecone not configured."
            return f"No search index available. {diag}"

    # Prefer to locate model in common locations (project model, parent-level model folders)
    model_basename = os.path.basename(MODEL_PATH)
    model_candidates = [
        MODEL_PATH,
        os.path.join(os.getcwd(), "model", model_basename),
        os.path.join(os.getcwd(), "..", "model", model_basename),
        os.path.join(os.path.abspath(os.path.join(os.getcwd(), "..", "..")), "model", model_basename)
    ]

    found_model = None
    for p in model_candidates:
        try:
            if os.path.exists(p):
                found_model = os.path.abspath(p)
                break
        except Exception:
            continue

    if found_model:
        try:
            from langchain_community.llms import CTransformers
            llm = CTransformers(model=found_model, model_type="llama", config={"max_new_tokens": 512, "temperature": 0.8})
            prompt = f"Use the following context to answer the question.\n\nContext:\n{combined}\n\nQuestion: {query}\n\nAnswer:"
            # Use the LLM's invoke() API which returns a string
            out = llm.invoke(prompt)
            if isinstance(out, str):
                return out
            # Some LLM adapters may return structured objects
            elif isinstance(out, dict) and "text" in out:
                return out["text"]
            # Fallback to string conversion
            else:
                return str(out)
        except Exception as e:
            # Return helpful diagnostic message so the user can see why local model failed to produce an answer
            short_err = repr(e)
            return (
                f"Found a local ggml model at {found_model} but failed to load it: {short_err}\n\n"
                f"Retrieved context:\n\n{combined}\n\n"
                f"Tip: ensure `ctransformers` is installed in the .venv, the file is a valid ggml file, and you have enough memory to load it."
            )

    return (
        f"Retrieved context:\n\n{combined}\n\n"
        f"Note: no local LLM found to generate a concise answer. Place a compatible ggml model at {MODEL_PATH} or in a `model/` folder one or two levels up, or hook an external LLM."
    )
