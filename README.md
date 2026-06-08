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
│   ├── corpus/
├── docs/
├── evals/
├── notebooks/
├── src/
├── tests/
├── .gitignore
└── README.md
```

## Corpus


Rodar a CLI:
```bash
python -m astronomia_rag.cli
```

<!-- - Domínio: Astronomia observacional
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (fixo pelo curso)
- Vector store: FAISS
- LLM: Mistral-7B-Instruct Q4_K_M via Ollama
- Framework: LangChain (bônus)
- Memória: Histórico de conversa (bônus): `ConversationBufferMemory`
- Interface: CLI + Jupyter notebook de demo
- Idioma do corpus: Português -->