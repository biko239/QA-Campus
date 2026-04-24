import json
import os
import random
import re
import subprocess
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_AI_DIR = r"C:\Users\alian.ALI\Desktop\ai"
MYSQL_URI = os.environ.get("FYP_MYSQL_URI", "root:root@localhost:3306/fyp_db")


def mysql_rows(query):
    result = subprocess.run(
        ["mysqlsh", "--json=raw", "--sql", MYSQL_URI, "-e", query],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    rows = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue

        payload = json.loads(line)
        if "rows" in payload:
            rows.extend(payload["rows"])

    return rows


def clean_text(value):
    return re.sub(r"\s+", " ", value or "").strip()


def tokenize(value):
    stop = {
        "the", "and", "for", "are", "you", "your", "with", "that", "this", "what",
        "when", "where", "who", "why", "how", "can", "could", "should", "would",
        "about", "into", "from", "does", "have", "has", "had", "will", "shall",
        "document", "documents", "policy", "university", "student", "students",
    }
    return [
        token
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", value.lower())
        if token not in stop
    ]


def keywords(row):
    text = clean_text(" ".join([
        row.get("Title", ""),
        row.get("Department", ""),
        row.get("Course", ""),
        row.get("Category", ""),
        row.get("Tags", ""),
        row.get("Text", ""),
    ]))
    words = tokenize(text)
    seen = []
    for word in words:
        if word not in seen:
            seen.append(word)
    return seen[:5] or [clean_text(row.get("Title", "document"))]


def make_questions(row):
    title = clean_text(row.get("Title", "the document"))
    category = clean_text(row.get("Category", "this topic"))
    key = keywords(row)[0]
    return [
        f"What does {title} say about {key}?",
        f"Explain the {category} information in {title}.",
        f"What should I know from {title} about {key}?",
    ]


def preview(text, limit=800):
    text = clean_text(text)
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def build_generator_user(question, row):
    return f"""Use only the source below to answer the question.

Question:
{question}

Source 1:
Document: {clean_text(row.get("Title", "Unknown Document"))} | Chunk ID: {row["Id"]}
{preview(row.get("Text", ""), 1200)}

Answer:"""


def build_generator_answer(row):
    title = clean_text(row.get("Title", "the uploaded document"))
    return (
        "According to the uploaded document, the relevant information is:\n\n"
        f"{preview(row.get('Text', ''), 700)}\n\n"
        "Sources: [Source 1]"
    )


def main():
    ai_dir = os.environ.get("FYP_AI_DIR", DEFAULT_AI_DIR)
    data_dir = os.path.join(ai_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    chunks = mysql_rows(
        """
        SELECT c.Id, c.Text, c.ChunkIndex,
               d.Id AS DocumentId, d.Title, d.Department, d.Course, d.Category, d.Tags
        FROM DocumentChunks c
        JOIN Documents d ON d.Id = c.DocumentId
        WHERE d.Status = 'READY' AND c.Text IS NOT NULL AND LENGTH(c.Text) > 80
        ORDER BY d.Id, c.ChunkIndex;
        """
    )

    if len(chunks) < 2:
        raise RuntimeError("Need at least 2 chunks to create positive/negative training pairs.")

    retriever_examples = []
    generator_examples = []
    random.seed(7)

    for index, row in enumerate(chunks):
        positive = clean_text(row["Text"])
        negative_choices = [candidate for candidate in chunks if candidate["Id"] != row["Id"]]
        negative = clean_text(random.choice(negative_choices)["Text"])

        for question in make_questions(row):
            retriever_examples.append({
                "question": question,
                "positive_chunk": positive,
                "negative_chunk": negative,
                "document_title": clean_text(row.get("Title", "")),
                "chunk_id": row["Id"],
            })

            generator_examples.append({
                "messages": [
                    {"role": "system", "content": "You are a helpful university assistant. Use only the provided source and cite it."},
                    {"role": "user", "content": build_generator_user(question, row)},
                    {"role": "assistant", "content": build_generator_answer(row)},
                ],
                "chunk_id": row["Id"],
                "document_title": clean_text(row.get("Title", "")),
            })

    retriever_path = os.path.join(data_dir, "retriever_train.json")
    generator_path = os.path.join(data_dir, "generator_train.jsonl")
    manifest_path = os.path.join(data_dir, "training_manifest.json")

    with open(retriever_path, "w", encoding="utf-8") as f:
        json.dump(retriever_examples, f, ensure_ascii=False, indent=2)

    with open(generator_path, "w", encoding="utf-8") as f:
        for item in generator_examples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "chunk_count": len(chunks),
        "retriever_example_count": len(retriever_examples),
        "generator_example_count": len(generator_examples),
        "retriever_path": retriever_path,
        "generator_path": generator_path,
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
