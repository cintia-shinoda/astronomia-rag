"""
Indexador: lê o corpus, divide em chunks, gera embeddings, salva em FAISS.

Este script roda UMA VEZ (offline). Depois disso, o índice fica em disco
e é reutilizado pelas consultas.

Uso:
    python -m astronomia_rag.indexer
"""

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from .config import (
    DATA_DIR,
    INDEX_DIR,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


def load_corpus():
    """
    Lê todos os arquivos .txt do diretório do corpus.

    Retorna uma lista de Documents (estrutura do LangChain
    contendo .page_content e .metadata).
    """
    loader = DirectoryLoader(
        str(DATA_DIR),
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    documents = loader.load()
    print(f"Carregados {len(documents)} documentos do corpus.")
    return documents


def split_documents(documents):
    """
    Divide cada documento em chunks menores.

    Por que `RecursiveCharacterTextSplitter`?
    Ele tenta quebrar primeiro em parágrafos ("\\n\\n"), depois em
    linhas ("\\n"), depois em frases (". "), depois em palavras (" "),
    depois caractere a caractere. Isso preserva a estrutura semântica
    melhor que um split fixo no caractere N.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,  # mede em caracteres (não tokens)
    )
    chunks = splitter.split_documents(documents)
    print(f"Gerados {len(chunks)} chunks "
          f"(média de {sum(len(c.page_content) for c in chunks) // len(chunks)} chars cada).")
    return chunks


def build_index(chunks):
    """
    Gera embeddings de cada chunk e indexa no FAISS.

    `normalize_embeddings=True` faz a similaridade do cosseno virar
    produto interno simples — FAISS lida com isso mais rápido.
    """
    print(f"Carregando embedder ({EMBEDDING_MODEL})...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        # No Mac M3 Pro, "mps" usa a GPU integrada (Apple Silicon).
        # Em outras máquinas, use "cpu" (ou "cuda" se tiver GPU NVIDIA).
        model_kwargs={"device": "mps"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print(f"Gerando embeddings e indexando {len(chunks)} chunks no FAISS...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def main():
    """Pipeline completo de indexação."""
    documents = load_corpus()
    if not documents:
        raise SystemExit(
            f"Nenhum .txt encontrado em {DATA_DIR}. "
            f"Crie o corpus primeiro."
        )

    chunks = split_documents(documents)
    vectorstore = build_index(chunks)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(INDEX_DIR))
    print(f"Índice FAISS salvo em {INDEX_DIR}.")
    print("Pronto! Agora rode `python -m astronomia_rag.cli` para conversar.")


if __name__ == "__main__":
    main()