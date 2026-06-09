
# Interface web do Astronomia RAG, usando Chainlit.


import chainlit as cl

from astronomia_rag.chain import build_chain
from astronomia_rag.memory import ChatHistory


# ============================================================
# Perguntas-exemplo (aparecem como botões clicáveis na tela inicial)
# ------------------------------------------------------------
# Para um leigo, a tela em branco é intimidante. Os "starters" dão
# pontos de partida — todos com resposta garantida no corpus.
# ============================================================
@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Por que vemos sempre a mesma face da Lua?",
            message="Por que sempre vemos a mesma face da Lua?",
        ),
        cl.Starter(
            label="O que é uma constelação?",
            message="O que é uma constelação?",
        ),
        cl.Starter(
            label="Dá para observar planetas na cidade?",
            message="É possível observar planetas de dentro de uma cidade grande?",
        ),
        cl.Starter(
            label="O que é magnitude aparente?",
            message="O que é magnitude aparente de uma estrela?",
        ),
    ]


# ============================================================
# Início da sessão: carrega o motor RAG uma vez por usuário
# ============================================================
@cl.on_chat_start
async def on_chat_start():
    # build_chain carrega o índice FAISS e conecta no Ollama
    chain = build_chain()
    cl.user_session.set("chain", chain)
    cl.user_session.set("history", ChatHistory())


# ============================================================
# A cada pergunta: recupera, gera (com streaming) e mostra as fontes
# ============================================================
@cl.on_message
async def on_message(message: cl.Message):
    chain = cl.user_session.get("chain")
    history = cl.user_session.get("history")
    question = message.content

    # Mensagem que vai sendo preenchida token a token
    answer_msg = cl.Message(content="")
    await answer_msg.send()

    source_docs = []

    # astream emite pedaços do resultado: a chave "context" traz os chunks recuperados; a chave "answer" traz a resposta em pedaços.
    async for chunk in chain.astream(
        {"input": question, "chat_history": history.get_messages()}
    ):
        if "context" in chunk:
            source_docs = chunk["context"]
        if "answer" in chunk:
            await answer_msg.stream_token(chunk["answer"])

    # Anexa as fontes consultadas
    if source_docs:
        seen: dict[str, int] = {}
        elements = []
        for doc in source_docs:
            fname = doc.metadata.get("source", "?").split("/")[-1]
            seen[fname] = seen.get(fname, 0) + 1
            label = fname if seen[fname] == 1 else f"{fname} ({seen[fname]})"
            snippet = doc.page_content[:140].replace("\n", " ").strip()
            elements.append(cl.Text(name=label, content=snippet, display="inline"))
        answer_msg.elements = elements

    await answer_msg.update()

    # Atualiza o histórico para perguntas de acompanhamento
    history.add_user_message(question)
    history.add_ai_message(answer_msg.content)