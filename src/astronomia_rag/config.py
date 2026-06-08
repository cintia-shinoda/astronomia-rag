"""
Configurações centrais do projeto.

Tudo que pode ser ajustado experimentalmente (chunk size, top_k, modelos)
Facilita rodar experimentos sem caçar parâmetros pelo código.
"""

from pathlib import Path

# ============================================================
# Caminhos
# ============================================================
# __file__ = .../src/astronomia_rag/config.py
# .parent.parent.parent = raiz do projeto
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "corpus"
INDEX_DIR = PROJECT_ROOT / "faiss_index"

# ============================================================
# Modelos
# ============================================================
# Embedder: o que transforma texto em vetor.
# all-MiniLM-L6-v2 = 384 dimensões, ~80MB, rápido.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# LLM: o que gera a resposta final.
# "mistral" é a tag do Ollama. Após `ollama pull mistral`,
# fica disponível como "mistral:latest".
LLM_MODEL = "mistral"

# ============================================================
# Hyperparâmetros do RAG
# ============================================================
# Tamanho do chunk em caracteres (não tokens).
# 500 chars ≈ 80-100 palavras em português.
CHUNK_SIZE = 500

# Overlap entre chunks adjacentes, em caracteres.
# Mantém contexto que está na fronteira entre dois chunks.
CHUNK_OVERLAP = 50

# Quantos chunks recuperar para responder cada pergunta.
# 3 é um bom default: contexto suficiente sem encher o LLM de ruído.
TOP_K = 3

# Temperatura do LLM: 0 = determinístico, 1 = criativo.
# Para FAQ factual queremos respostas estáveis: 0.2-0.3.
LLM_TEMPERATURE = 0.1