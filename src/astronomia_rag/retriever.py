"""
Retriever: carrega o índice FAISS e busca chunks relevantes.

Isolar o retriever num módulo facilita testar (você pode testar a
recuperação sem rodar o LLM).
"""

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from .config import INDEX_DIR, EMBEDDING_MODEL, TOP_K


# Cache global do embedder: evita recarregar o modelo (~80MB) toda vez.
_embeddings_cache = None


def get_embeddings():
    """Cria (ou reaproveita) o embedder. Mesmo modelo do indexer."""
    global _embeddings_cache
    if _embeddings_cache is None:
        _embeddings_cache = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "mps"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings_cache


def load_retriever(top_k: int = TOP_K):
    """
    Carrega o índice FAISS do disco e devolve um retriever.

    `allow_dangerous_deserialization=True` é necessário porque o FAISS
    serializa via pickle. Seguro se você gerou o índice (não baixou
    de fonte desconhecida).
    """
    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(
        str(INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectorstore.as_retriever(search_kwargs={"k": top_k})


def search(query: str, top_k: int = TOP_K):
    """
    Busca direta sem chain — útil pra debugar a recuperação.

    Exemplo:
        from astronomia_rag.retriever import search
        docs = search("o que é magnitude aparente?")
        for d in docs:
            print(d.metadata["source"], "→", d.page_content[:100])
    """
    retriever = load_retriever(top_k=top_k)
    return retriever.invoke(query)