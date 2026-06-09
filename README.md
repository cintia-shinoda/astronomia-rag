# `Astronomia RAG`

<p align="center">
  <img src= "https://img.shields.io/badge/status-in%20progress-yellow" alt="Status do Projeto" />
  <img src="https://img.shields.io/github/last-commit/cintia-shinoda/astronomia-rag" alt="GitHub Last Commit" />
  <img src="https://img.shields.io/github/forks/cintia-shinoda/astronomia-rag" alt="GitHub Forks" />
  <img src="https://img.shields.io/github/stars/cintia-shinoda/astronomia-rag" alt="GitHub Stars" />
</p>

<br>


> Sistema de perguntas e respostas (FAQ) sobre Astronomia Observacional baseado em Retrieval-Augmented Generation (`RAG`), com busca vetorial em `FAISS`, LLM local (`Mistral-7B`) e pipeline de processamento construído com `LangChain`.


---


## Estrutura do Repositório

```bash
astronomia-rag/
├── data/
│   └── corpus/
│
├── docs/
│
├── evals/
│   ├── eval_results.json
│   ├── golden_set.json
│   ├── preencher_notas.py
│   └── run_evals.py
│
├── notebooks/
│
├── src/astronomia_rag/
│   ├── __init__.py
│   ├── chain.py
│   ├── cli.py
│   ├── config.py
│   ├── indexer.py
│   ├── memory.py
│   └── retriever.py
│
├── tests/
│   └── test_retriever.py
│
├── .gitignore
├── cli_demo.svg
├── pyproject.toml
└── README.md
```

## Corpus
Os 15 documentos do corpus foram redigidos como material educacional conciso, com auxílio de IA, a partir de conhecimento consolidado de Astronomia.


### Como executar:
#### Requisitos:
- Python 3.10+
- Ollama

1. Clone o repositório e instale as dependências:
```bash
git clone https://github.com/cintia-shinoda/astronomia-rag.git

cd astronomia-rag

pip install -e ".[dev]"
```

2. Download do modelo via Ollama:
```bash
ollama pull mistral
```

3. Rodar o pipeline de RAG via CLI:
```bash
python -m astronomia_rag.indexer
python -m astronomia_rag.cli
```