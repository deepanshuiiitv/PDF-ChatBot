import os
from dotenv import load_dotenv
load_dotenv()

VECTFILE = os.path.join(os.getcwd(), "vectorstore.pkl")


def clear_index(delete_index: bool = False):
    """Clear local vectorstore, in-memory RAG data and optionally clear Pinecone index.

    Returns a dict with `messages` explaining what was done or failed.
    """
    messages = []

    # remove persisted vectorstore
    try:
        if os.path.exists(VECTFILE):
            os.remove(VECTFILE)
            messages.append("Removed local vectorstore file")
        else:
            messages.append("No local vectorstore found")
    except Exception as e:
        messages.append(f"Failed to remove local vectorstore: {e}")

    # try to clear in-memory structures
    try:
        import rag_engine
        rag_engine.DOCUMENTS = []
        rag_engine.EMBEDDINGS = None
        rag_engine.NN = None
        messages.append("Cleared in-memory RAG stores")
    except Exception as e:
        messages.append(f"Failed to clear in-memory RAG stores: {e}")

    # If Pinecone is configured, try to clear its index
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX")
    pine_env = os.getenv("PINECONE_ENV") or os.getenv("PINECONE_ENVIRONMENT")

    if api_key and index_name:
        try:
            # Newer pinecone package exposes a Pinecone class instead of a top-level init()
            try:
                from pinecone import Pinecone
                pc = Pinecone(api_key=api_key)
                idx = pc.Index(index_name)
                idx.delete(delete_all=True)
                messages.append(f"Cleared Pinecone index '{index_name}' via Pinecone client")
                if delete_index:
                    try:
                        pc.delete_index(index_name)
                        messages.append(f"Deleted Pinecone index '{index_name}'")
                    except Exception as e:
                        messages.append(f"Failed to delete Pinecone index: {e}")
            except Exception:
                # Fallback for older packages providing init()
                import pinecone
                if pine_env:
                    pinecone.init(api_key=api_key, environment=pine_env)
                else:
                    pinecone.init(api_key=api_key)
                idx = pinecone.Index(index_name)
                idx.delete(delete_all=True)
                messages.append(f"Cleared Pinecone index '{index_name}' via legacy pinecone.init()")
                if delete_index:
                    try:
                        pinecone.delete_index(index_name)
                        messages.append(f"Deleted Pinecone index '{index_name}'")
                    except Exception as e:
                        messages.append(f"Failed to delete Pinecone index: {e}")
        except Exception as e:
            messages.append(f"Failed to operate on Pinecone index '{index_name}': {e}")
    else:
        messages.append("Pinecone not configured or missing env vars; skipped Pinecone operations")

    return {"messages": messages}
