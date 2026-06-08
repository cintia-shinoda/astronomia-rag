"""
Interface de linha de comando.

Uso:
    python -m astronomia_rag.cli
"""

from .chain import build_chain
from .memory import ChatHistory


BANNER = """
╔══════════════════════════════════════════════════════════════╗
║  🔭  Astronomia RAG — pergunte sobre o céu                   ║
║                                                              ║
║  Comandos: 'sair' para sair, 'limpar' para limpar histórico  ║
╚══════════════════════════════════════════════════════════════╝
"""


def format_sources(source_docs) -> str:
    """Formata os documentos recuperados pra exibir como 'fontes'."""
    if not source_docs:
        return ""

    lines = [f"\nFontes consultadas ({len(source_docs)} chunks):"]
    for i, doc in enumerate(source_docs, 1):
        # Pega só o nome do arquivo, não o caminho completo
        source_file = doc.metadata.get("source", "?").split("/")[-1]
        # Primeiros 100 chars do chunk pra dar uma ideia do trecho
        snippet = doc.page_content[:100].replace("\n", " ").strip()
        lines.append(f"  {i}. {source_file}: {snippet}...")
    return "\n".join(lines)


def main():
    print(BANNER)
    print("Carregando modelos (primeira vez pode demorar)...")

    chain = build_chain()
    history = ChatHistory()

    print("Pronto! Faça sua pergunta.\n")

    while True:
        try:
            question = input("❓ Você: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nAté logo!")
            break

        if not question:
            continue
        if question.lower() in ("sair", "exit", "quit"):
            print("Até logo!")
            break
        if question.lower() in ("limpar", "clear"):
            history.clear()
            print("Histórico limpo.\n")
            continue

        # Roda a chain
        response = chain.invoke({
            "input": question,
            "chat_history": history.get_messages(),
        })

        answer = response["answer"]
        sources = response.get("context", [])

        print(f"\nBot: {answer}")
        print(format_sources(sources))
        print()

        # Atualiza histórico
        history.add_user_message(question)
        history.add_ai_message(answer)


if __name__ == "__main__":
    main()