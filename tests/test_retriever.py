from astronomia_rag.retriever import search

pergunta = "o que é magnitude aparente?"
print(f"Pergunta: {pergunta}\n")

for d in search(pergunta):
    arquivo = d.metadata["source"].split("/")[-1]
    trecho = d.page_content[:80].replace("\n", " ")
    print(f"{arquivo}  →  {trecho}")