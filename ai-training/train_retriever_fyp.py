import json
import os
import shutil
from datetime import datetime

from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_AI_DIR = r"C:\Users\alian.ALI\Desktop\ai"


def main():
    ai_dir = os.environ.get("FYP_AI_DIR", DEFAULT_AI_DIR)
    data_path = os.path.join(ai_dir, "data", "retriever_train.json")
    model_path = os.path.join(ai_dir, "models", "retriever")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(ai_dir, "models", f"retriever_backup_{stamp}")
    trained_path = os.path.join(ai_dir, "models", f"retriever_trained_{stamp}")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data not found: {data_path}")

    shutil.copytree(model_path, backup_path)
    print(f"Backed up retriever to: {backup_path}")

    model = SentenceTransformer(model_path)

    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    train_examples = []
    for item in raw_data:
        question = item["question"].strip()
        positive = item["positive_chunk"].strip()
        negative = item["negative_chunk"].strip()
        train_examples.append(InputExample(texts=[question, positive], label=1.0))
        train_examples.append(InputExample(texts=[question, negative], label=0.0))

    dataloader = DataLoader(train_examples, shuffle=True, batch_size=4)
    train_loss = losses.CosineSimilarityLoss(model)

    model.fit(
        train_objectives=[(dataloader, train_loss)],
        epochs=int(os.environ.get("FYP_RETRIEVER_EPOCHS", "2")),
        warmup_steps=5,
        output_path=trained_path,
        show_progress_bar=True,
    )

    print(f"Retriever training complete. Saved trained model to: {trained_path}")
    print(f"Original active model is still at: {model_path}")
    print("After this process exits, replace the active retriever folder with the trained folder.")


if __name__ == "__main__":
    main()
