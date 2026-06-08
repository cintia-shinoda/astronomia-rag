"""
Gerenciamento simples de histórico de conversa.

Em LCEL, o histórico é passado como uma lista de Messages
(HumanMessage, AIMessage). Esta classe gerencia essa lista.

Poderia ser substituído por ConversationBufferMemory, mas explícito
é mais didático e mais flexível.
"""

from langchain_core.messages import HumanMessage, AIMessage


class ChatHistory:
    """Histórico de conversa como lista de Messages."""

    def __init__(self):
        self.messages: list = []

    def add_user_message(self, content: str) -> None:
        self.messages.append(HumanMessage(content=content))

    def add_ai_message(self, content: str) -> None:
        self.messages.append(AIMessage(content=content))

    def clear(self) -> None:
        self.messages = []

    def get_messages(self) -> list:
        return self.messages

    def __len__(self) -> int:
        return len(self.messages)