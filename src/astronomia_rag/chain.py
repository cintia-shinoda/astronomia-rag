"""
Chain RAG completa usando LCEL.

Fluxo:
1. Pergunta entra com histórico de conversa.
2. Um sub-chain reformula a pergunta para ser autossuficiente
   (importante: "ele tem luas?" depois de "fale de Júpiter" precisa
   virar "Júpiter tem luas?" antes da busca FAISS).
3. Retriever busca top-k chunks.
4. Prompt final monta contexto + histórico + pergunta.
5. Mistral-7B gera resposta.
"""

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from .config import LLM_MODEL, LLM_TEMPERATURE
from .retriever import load_retriever


# ============================================================
# Prompt 1: reformulação da pergunta com base no histórico
# ============================================================
# Sem isso, perguntas como "ele tem luas?" gerariam embedding ruim
# (sem contexto, "ele" não diz nada). Esse sub-chain "condensa" a
# pergunta usando o histórico.
CONDENSE_QUESTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Dado o histórico da conversa e a última pergunta do usuário, "
     "reformule a pergunta para que seja autossuficiente "
     "(possa ser entendida sem o histórico). "
     "Se já for autossuficiente, retorne-a como está. "
     "Responda APENAS com a pergunta reformulada, sem explicações."),
    MessagesPlaceholder("chat_history"),
    ("user", "{input}"),
])


# ============================================================
# Prompt 2: geração da resposta final
# ============================================================
# Este é o prompt principal. `{context}` é preenchido pelo
# `create_stuff_documents_chain` com os chunks recuperados.
QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Você responde perguntas sobre astronomia usando EXCLUSIVAMENTE o contexto "
     "fornecido abaixo. Você NÃO tem permissão para usar conhecimento próprio.\n\n"
     "REGRAS (siga sem exceção):\n"
     "1. Responda SOMENTE com base no contexto abaixo.\n"
     "2. Se a informação NÃO estiver no contexto, responda apenas:\n"
     "   'Não encontrei essa informação no meu material sobre astronomia.'\n"
     "   Não escreva mais nada depois disso.\n"
     "3. A regra 2 vale MESMO que você saiba a resposta de outra fonte. "
     "Capital de países, receitas, esportes — se não está no contexto, recuse.\n"
     "4. Nunca invente números, datas ou nomes ausentes do contexto.\n"
     "5. Seja claro e conciso (2 a 5 frases).\n\n"
     "EXEMPLO 1:\n"
     "Pergunta: Qual a capital da França?\n"
     "Resposta correta: Não encontrei essa informação no meu material sobre astronomia.\n\n"
     "EXEMPLO 2:\n"
     "Pergunta: Me dê uma receita de bolo.\n"
     "Resposta correta: Não encontrei essa informação no meu material sobre astronomia.\n\n"
     "Use o contexto abaixo para responder.\n\n"
     "Contexto:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("user", "{input}"),
])


def build_chain():
    """
    Monta a chain RAG completa.

    Retorna um objeto que aceita .invoke({"input": ..., "chat_history": [...]})
    e devolve {"answer": ..., "context": [docs...]}.
    """
    # LLM via Ollama
    llm = OllamaLLM(model=LLM_MODEL, temperature=LLM_TEMPERATURE)

    # Retriever (carrega FAISS do disco)
    retriever = load_retriever()

    # Sub-chain 1: retriever que considera histórico para reformular
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, CONDENSE_QUESTION_PROMPT
    )

    # Sub-chain 2: combina os documentos recuperados num prompt
    # ("stuff" = "enfia tudo" no contexto; ok pra top_k pequeno)
    qa_chain = create_stuff_documents_chain(llm, QA_PROMPT)

    # Chain final: retrieve → generate
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    return rag_chain