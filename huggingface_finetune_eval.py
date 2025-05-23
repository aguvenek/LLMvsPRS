import os
import json
import gc
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedModel
from datasets import load_dataset
import torch.nn as nn
import torch.nn.functional as F

from standalone_hyenadna import HyenaDNAModel
from standalone_hyenadna import CharacterTokenizer
from huggingface_modified import HyenaDNAPreTrainedModel

# Configs
file_path = "/sc/arion/projects/mscic1/PRS-LLM/data/clumped/ayub_clumped_samples_sequences_40samples_with_labels_withSNP.csv"
cache_dir = "/sc/arion/projects/mscic1/PRS-LLM/hf_cache"
pretrained_model_name = "hyenadna-large-1m-seqlen"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load dataset
print("Loading dataset...")
dataset = load_dataset(
    "csv",
    data_files=file_path,
    split="train",
    cache_dir=cache_dir
)
print(f"Loaded dataset with {len(dataset)} samples")

# Load tokenizer
max_length = 1_000_000

print("Initializing tokenizer...")
tokenizer = CharacterTokenizer(
    characters=['A', 'C', 'G', 'T', 'N'],
    model_max_length=max_length + 2,
    add_special_tokens=False,
    padding_side='left'
)

# Load pretrained model
print("Loading model...")
model = HyenaDNAPreTrainedModel.from_pretrained(
    './checkpoints',
    pretrained_model_name,
    download=True,
    config=None,
    device=device,
    use_head=False,
    n_classes=2
)
model.to(device)
model.train()

# Classifier head
n_classes = 2
embedding_dim = 256
classifier = nn.Linear(embedding_dim, n_classes).to(device)

# Optimizer and loss
optimizer = torch.optim.AdamW(list(model.parameters()) + list(classifier.parameters()), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()

# Training loop
print("Starting fine-tuning...")
epochs = 1  # increase as needed
for epoch in range(epochs):
    total_loss = 0
    for i in range(len(dataset)):
        sequence = dataset[i]["sequence"][:200_000]  # trimmed to avoid OOM
        label = dataset[i]["labels"]

        tok = tokenizer(sequence)["input_ids"]
        tok = torch.LongTensor(tok).unsqueeze(0).to(device)
        label_tensor = torch.tensor([label]).to(device)

        optimizer.zero_grad()

        with torch.autocast(device_type='cuda', enabled=False):  # no mixed precision
            embeddings = model(tok)
            logits = classifier(embeddings[:, -1, :])
            loss = loss_fn(logits, label_tensor)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Memory cleanup
        torch.cuda.empty_cache()
        gc.collect()

        print(f"Sample {i+1}/{len(dataset)} | Loss: {loss.item():.4f} | Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    print(f"Epoch {epoch+1} finished. Average Loss: {total_loss/len(dataset):.4f}")

# Save model
save_path = "./checkpoints/hyenadna_finetuned.pt"
torch.save({
    'model_state_dict': model.state_dict(),
    'classifier_state_dict': classifier.state_dict(),
    'tokenizer': tokenizer,
}, save_path)
print(f"Model saved to {save_path}")



# ----------- Evaluation after fine-tuning -----------

### [Added] Import and setup
import evaluate
metric = evaluate.combine([
    evaluate.load("precision"),
    evaluate.load("recall"),
    evaluate.load("f1")
])

model.eval()
classifier.eval()

predictions = []
true_labels = []

print("Running evaluation...")
for i in range(len(dataset)):
    sequence = dataset[i]["sequence"][:200_000]
    true_label = dataset[i]["labels"]

    tok = tokenizer(sequence)["input_ids"]
    tok = torch.LongTensor(tok).unsqueeze(0).to(device)

    with torch.inference_mode():
        embeddings = model(tok)
        logits = classifier(embeddings[:, -1, :])
        pred = torch.argmax(logits, dim=-1).item()

    predictions.append(pred)
    true_labels.append(true_label)

results = metric.compute(predictions=predictions, references=true_labels)
print("Evaluation Results:", results)

