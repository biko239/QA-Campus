# Local AI Setup

The chatbot should not be fine-tuned on the currently uploaded PDFs. Uploaded PDFs are handled through RAG:

1. extract text,
2. split into chunks,
3. embed/index chunks,
4. retrieve the most relevant chunks for each question,
5. let the clean local generator answer only from those chunks.

This keeps the assistant useful for future PDFs instead of overfitting to today's files.

## Reset Bad Training

Run this when the local AI has been over-trained or starts answering poorly:

```powershell
cd C:\Users\alian.ALI\Desktop\fyp\fyp_fixed\ai-training
python reset_local_ai.py
```

This removes:

- `models/generator_adapter`
- trained/fine-tuned retriever folders and backups
- generated PDF training data
- training checkpoints

It keeps the base generator model and downloads a clean general retriever into:

```text
C:\Users\alian.ALI\Desktop\ai\models\retriever
```

## Advanced

The old training scripts are kept for experimentation only. Do not use them for normal project work unless you have a large, high-quality, general dataset. For this app, future PDFs should be uploaded and indexed, not baked into model weights.
