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
    text = value or ""
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    text = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", text)
    text = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", text)
    return re.sub(r"\s+", " ", text).strip()


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
    text = clean_text(row.get("Text", ""))
    questions = [
        f"What does {title} say about {key}?",
        f"Explain the {category} information in {title}.",
        f"What should I know from {title} about {key}?",
        f"Give me the important details from {title}.",
        f"Answer naturally using {title}: what is important here?",
        f"What rules, dates, responsibilities, or procedures are mentioned in {title}?",
    ]

    lowered = text.lower()
    if any(term in lowered for term in ["semester", "semestre", "debut sem", "examens", "exams"]):
        questions.extend([
            "Give me the complete semester timeline.",
            "When does semester 1 start and end?",
            "When does semester 2 start and end?",
            "What are the important exam dates?",
            "List the important academic calendar dates.",
        ])

    if any(term in lowered for term in ["article", "section", "policy", "procedure", "responsibility"]):
        questions.extend([
            f"What procedure is described in {title}?",
            f"What responsibilities are described in {title}?",
            f"Summarize the relevant article or section in {title}.",
        ])

    if any(term in lowered for term in ["request", "access", "approval", "retention", "privacy", "security"]):
        questions.extend([
            f"How are requests handled in {title}?",
            f"What privacy or security rule is mentioned in {title}?",
            f"What retention or approval rule is mentioned in {title}?",
        ])

    return questions


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

Write a natural answer with only the source above."""


def build_generator_answer(row):
    title = clean_text(row.get("Title", "the uploaded document"))
    return (
        f"Based on {title}, the relevant answer is:\n\n"
        f"{preview(row.get('Text', ''), 700)}\n\n"
        "Sources: [Source 1]"
    )


def is_calendar_row(row):
    metadata = " ".join([
        row.get("Title", ""),
        row.get("Category", ""),
        row.get("Tags", ""),
    ]).lower()
    return "calendar" in metadata or "calendrier" in metadata


def is_study_regulations_row(row):
    title = clean_text(row.get("Title", "")).lower()
    return "study regulation" in title or "reglement" in title or "règlement" in title


def calendar_hard_questions():
    return [
        "sem 1 exams",
        "semestre 1 exams",
        "semester 1 exams",
        "exams in sem 1",
        "exams in semestre 1",
        "examens sem 1",
        "examens semestre 1",
        "sem 2 exams",
        "semester 2 exams",
        "exams in sem 2",
        "examens semestre 2",
        "when are sem 1 exams",
        "when are semester 1 exams",
        "when are sem 2 exams",
        "when are semester 2 exams",
        "sem 1 dates",
        "sem 2 dates",
        "semester 1 timeline",
        "semester 2 timeline",
    ]


def calendar_answer_for_question(question):
    q = question.lower()
    if "sem 2" in q or "semester 2" in q or "semestre 2" in q:
        return (
            "Semester 2 exam dates in the calendar are:\n\n"
            "- May 4-8, 2026: Examens\n"
            "- May 9, 2026: Langues\n"
            "- May 11-16, 2026: Examens\n\n"
            "Sources: [Source 1]"
        )

    return (
        "Semester 1 exam dates in the calendar are:\n\n"
        "- December 8-12, 2025: Examens\n"
        "- December 13, 2025: Langues\n"
        "- December 15-20, 2025: Examens\n\n"
        "Sources: [Source 1]"
    )


def add_hard_calendar_examples(chunks, retriever_examples, generator_examples):
    calendar_rows = [row for row in chunks if is_calendar_row(row)]
    if not calendar_rows:
        return

    calendar_row = calendar_rows[0]
    positive = clean_text(calendar_row["Text"])
    hard_negative_rows = [row for row in chunks if is_study_regulations_row(row)]
    if not hard_negative_rows:
        hard_negative_rows = [row for row in chunks if row["Id"] != calendar_row["Id"]]

    for question in calendar_hard_questions():
        for negative_row in hard_negative_rows[:8]:
            retriever_examples.append({
                "question": question,
                "positive_chunk": positive,
                "negative_chunk": clean_text(negative_row["Text"]),
                "document_title": clean_text(calendar_row.get("Title", "")),
                "chunk_id": calendar_row["Id"],
                "hard_example": True,
            })

        generator_examples.append({
            "messages": [
                {"role": "system", "content": "You are a helpful university assistant. Use only the provided source and cite it."},
                {"role": "user", "content": build_generator_user(question, calendar_row)},
                {"role": "assistant", "content": calendar_answer_for_question(question)},
            ],
            "chunk_id": calendar_row["Id"],
            "document_title": clean_text(calendar_row.get("Title", "")),
            "hard_example": True,
        })


def transform_prompt(instruction, previous_answer):
    return f"""Rewrite the previous assistant answer according to the user's instruction.
Do not search for new information.
Do not add facts that are not present in the previous answer.

Instruction:
{instruction}

Previous answer:
{previous_answer}

Rewritten answer:"""


def add_transform_examples(generator_examples):
    examples = [
        (
            "Translate the previous answer to clear English.",
            "Les règles principales sont : l'année académique comporte deux semestres de 14 semaines, chaque semestre représente 30 crédits ECTS, et les examens finaux sont fixés par le calendrier universitaire.",
            "The main rules are: the academic year has two 14-week semesters, each semester represents 30 ECTS credits, and final exams are scheduled by the university calendar."
        ),
        (
            "Make the previous answer shorter while keeping the important facts.",
            "Semester 1 starts on September 1, 2025 and ends on January 18, 2026. The main exam dates are December 8-12, December 13 for languages, and December 15-20, 2025.",
            "Semester 1 runs from September 1, 2025 to January 18, 2026. Exams are December 8-13 and December 15-20, 2025."
        ),
        (
            "Explain the previous answer in simpler words.",
            "The credits are validated after assessment of acquired learning outcomes, and the jury may apply the rules defined by the institution.",
            "This means students earn credits only after being evaluated, and the institution's jury applies the official rules."
        ),
    ]

    for instruction, previous_answer, answer in examples:
        generator_examples.append({
            "messages": [
                {"role": "system", "content": "You rewrite previous assistant answers without adding new facts."},
                {"role": "user", "content": transform_prompt(instruction, previous_answer)},
                {"role": "assistant", "content": answer},
            ],
            "document_title": "conversation_follow_up",
            "chunk_id": 0,
            "hard_example": True,
        })


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

    add_hard_calendar_examples(chunks, retriever_examples, generator_examples)
    add_transform_examples(generator_examples)

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
