import os
import shutil
from pathlib import Path

from sentence_transformers import SentenceTransformer


DEFAULT_AI_DIR = Path(r"C:\Users\alian.ALI\Desktop\ai")
GENERAL_RETRIEVER = os.environ.get("FYP_GENERAL_RETRIEVER", "BAAI/bge-small-en-v1.5")


def remove_path(path: Path) -> None:
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        print(f"Removed: {path}")


def main() -> None:
    ai_dir = Path(os.environ.get("FYP_AI_DIR", DEFAULT_AI_DIR)).resolve()
    models_dir = ai_dir / "models"

    if not models_dir.exists():
        raise FileNotFoundError(f"AI models folder not found: {models_dir}")

    protected_generator = (models_dir / "generator").resolve()
    if not protected_generator.exists():
        raise FileNotFoundError(f"Base generator model is missing: {protected_generator}")

    cleanup_targets = [
        models_dir / "generator_adapter",
        models_dir / "retriever",
        ai_dir / "checkpoints",
        ai_dir / "data" / "generator_train.jsonl",
        ai_dir / "data" / "retriever_train.json",
        ai_dir / "data" / "training_manifest.json",
    ]

    cleanup_targets.extend(models_dir.glob("retriever_backup_*"))
    cleanup_targets.extend(models_dir.glob("retriever_before_*"))
    cleanup_targets.extend(models_dir.glob("retriever_failed_*"))
    cleanup_targets.extend(models_dir.glob("retriever_trained_*"))

    for target in cleanup_targets:
        resolved = target.resolve()
        if models_dir not in resolved.parents and ai_dir not in resolved.parents:
            raise RuntimeError(f"Refusing to remove path outside AI folder: {resolved}")
        if resolved == protected_generator:
            raise RuntimeError("Refusing to remove the base generator model.")
        remove_path(resolved)

    retriever_path = models_dir / "retriever"
    print(f"Downloading clean general retriever: {GENERAL_RETRIEVER}")
    model = SentenceTransformer(GENERAL_RETRIEVER)
    model.save(str(retriever_path))
    print(f"Saved clean retriever to: {retriever_path}")
    print("Done. The bad PDF-specific LoRA adapter and retriever training artifacts are gone.")


if __name__ == "__main__":
    main()
