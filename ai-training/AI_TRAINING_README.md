# AI Training Pipeline

This folder contains the scripts used to train the local AI service.

## Order

1. `export_training_data.py`
2. `train_retriever_fyp.py`
3. `train_generator_adapter.py`

The retriever is trained directly into `C:\Users\alian.ALI\Desktop\ai\models\retriever` after creating a timestamped backup.

The generator script trains a small LoRA adapter into:

```text
C:\Users\alian.ALI\Desktop\ai\models\generator_adapter
```

The full generator is not overwritten.
