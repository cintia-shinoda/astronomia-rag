import json
from pathlib import Path

# acha o eval_results.json na mesma pasta deste script (evals/)
EVAL_PATH = Path(__file__).parent / "eval_results.json"

# Notas (1-5) na ordem das 30 perguntas do golden_set
notas = [2, 2, 5, 4, 2, 2, 5, 4, 4, 2, 5, 5, 3, 4, 1,
         5, 5, 3, 5, 5, 5, 3, 3, 5, 4, 5, 2, 5, 5, 5]

with EVAL_PATH.open(encoding="utf-8") as f:
    data = json.load(f)

for item, nota in zip(data["generation"], notas):
    item["subjective_score"] = nota

with EVAL_PATH.open("w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

media = sum(notas) / len(notas)
print(f"Notas preenchidas. Média subjetiva: {media:.2f}/5")