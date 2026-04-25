import os
import re
from typing import List, Optional

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RETRIEVER_MODEL_PATH = os.path.join(BASE_DIR, "models", "retriever")
GENERATOR_MODEL_PATH = os.path.join(BASE_DIR, "models", "generator")
GENERATOR_ADAPTER_PATH = os.path.join(BASE_DIR, "models", "generator_adapter")
USE_GENERATOR_ADAPTER = os.environ.get("FYP_USE_GENERATOR_ADAPTER", "0") == "1"

app = FastAPI(title="USJ Local AI Service")

# -----------------------------
# Load retriever
# -----------------------------
retriever_model = SentenceTransformer(RETRIEVER_MODEL_PATH)

# -----------------------------
# Load generator
# -----------------------------
generator_tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL_PATH)
if generator_tokenizer.pad_token is None:
    generator_tokenizer.pad_token = generator_tokenizer.eos_token

device = "cuda" if torch.cuda.is_available() else "cpu"
generator_model = AutoModelForCausalLM.from_pretrained(
    GENERATOR_MODEL_PATH,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    low_cpu_mem_usage=True,
)

if USE_GENERATOR_ADAPTER and PeftModel is not None and os.path.isdir(GENERATOR_ADAPTER_PATH):
    generator_model = PeftModel.from_pretrained(generator_model, GENERATOR_ADAPTER_PATH)

generator_model.to(device)

# -----------------------------
# Schemas
# -----------------------------
class EmbedRequest(BaseModel):
    text: str
    mode: Optional[str] = "document"

class ChunkItem(BaseModel):
    chunk_id: int
    document_title: str
    text: str
    score: Optional[float] = None

class GenerateRequest(BaseModel):
    question: str
    chunks: List[ChunkItem]

class RerankRequest(BaseModel):
    question: str
    chunks: List[ChunkItem]

class TransformRequest(BaseModel):
    instruction: str
    text: str

# -----------------------------
# Small-talk handling
# -----------------------------
GREETING_PATTERNS = [
    r"^\s*hi\s*$",
    r"^\s*hello\s*$",
    r"^\s*hey\s*$",
    r"^\s*good morning\s*$",
    r"^\s*good evening\s*$",
]

THANKS_PATTERNS = [
    r"^\s*thanks\s*$",
    r"^\s*thank you\s*$",
    r"^\s*thank you so much\s*$",
]

BYE_PATTERNS = [
    r"^\s*bye\s*$",
    r"^\s*goodbye\s*$",
    r"^\s*see you\s*$",
]

def detect_smalltalk(text: str) -> Optional[str]:
    t = text.strip().lower()

    if t in {"thanks", "thank you", "thank you so much"}:
        return "You're welcome."

    for p in GREETING_PATTERNS:
        if re.match(p, t):
            return "Hello. How can I help you today?"
    for p in THANKS_PATTERNS:
        if re.match(p, t):
            return "You’re welcome."
    for p in BYE_PATTERNS:
        if re.match(p, t):
            return "Goodbye."
    return None

def build_prompt(question: str, chunks: List[ChunkItem]) -> str:
    context_blocks = []
    for i, ch in enumerate(chunks[:4], start=1):
        context_blocks.append(
            f"[Source {i}] Document: {ch.document_title} | Chunk ID: {ch.chunk_id}\n{ch.text}"
        )

    context = "\n\n".join(context_blocks)

    return f"""
You are a careful university document assistant.

Rules:
- Answer naturally and directly.
- Use ONLY the provided context.
- Do NOT invent facts.
- Prefer exact facts, names, dates, numbers, requirements, exceptions, and conditions from the sources.
- If the context is missing, weak, unrelated, or insufficient, answer exactly:
I could not find enough reliable information in the uploaded university documents.
- If sources conflict, say that the uploaded documents conflict and cite the relevant sources.
- Keep the answer organized, but do not add unnecessary decoration.
- At the end, write:
Sources: [Source numbers used]
- Do not mention any source that you did not use.

Question:
{question}

Context:
{context}

Answer:
""".strip()

def generate_text(prompt: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful university assistant."},
        {"role": "user", "content": prompt},
    ]

    text = generator_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = generator_tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = generator_model.generate(
            **inputs,
            max_new_tokens=360,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=generator_tokenizer.eos_token_id
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    answer = generator_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return answer

def retrieval_text(text: str, mode: Optional[str]) -> str:
    cleaned = (text or "").strip()
    if (mode or "").lower() == "query":
        return "query: " + cleaned
    return "passage: " + cleaned

# -----------------------------
# Endpoints
# -----------------------------
@app.get("/")
def home():
    return {"message": "USJ local AI service is running"}

@app.post("/embed")
def embed(req: EmbedRequest):
    vector = retriever_model.encode(
        retrieval_text(req.text, req.mode),
        normalize_embeddings=True
    ).tolist()
    return {"embedding": vector}

@app.post("/rerank")
def rerank(req: RerankRequest):
    if not req.chunks:
        return {"results": []}

    question_vector = retriever_model.encode(
        retrieval_text(req.question, "query"),
        normalize_embeddings=True
    )
    chunk_texts = [
        retrieval_text(f"Document: {chunk.document_title}\n{chunk.text}", "document")
        for chunk in req.chunks
    ]
    chunk_vectors = retriever_model.encode(chunk_texts, normalize_embeddings=True)
    scores = chunk_vectors @ question_vector

    results = []
    for chunk, score in zip(req.chunks, scores):
        results.append({
            "chunk_id": chunk.chunk_id,
            "score": float(score),
            "document_title": chunk.document_title
        })

    results.sort(key=lambda item: item["score"], reverse=True)
    return {"results": results}

@app.post("/transform")
def transform(req: TransformRequest):
    prompt = f"""
You are a helpful university assistant.

Rewrite the previous assistant answer according to the user's instruction.
Do not search for new information.
Do not add facts that are not present in the previous answer.
Keep the result natural and concise.

Instruction:
{req.instruction}

Previous answer:
{req.text}

Rewritten answer:
""".strip()

    answer = generate_text(prompt)
    if not answer:
        answer = req.text

    return {"answer": answer}

@app.post("/generate")
def generate(req: GenerateRequest):
    smalltalk = detect_smalltalk(req.question)
    if smalltalk is not None:
        return {
            "answer": smalltalk,
            "supported": True,
            "used_chunk_ids": []
        }

    if not req.chunks:
        return {
            "answer": "I could not find enough reliable information in the uploaded university documents.",
            "supported": False,
            "used_chunk_ids": []
        }

    prompt = build_prompt(req.question, req.chunks)
    answer = generate_text(prompt)

    if not answer:
        answer = "I could not find enough reliable information in the uploaded university documents."

    return {
        "answer": answer,
        "supported": "I could not find enough reliable information" not in answer,
        "used_chunk_ids": [c.chunk_id for c in req.chunks[:4]]
    }
