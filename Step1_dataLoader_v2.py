import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
# from datasets import load_dataset, Features, Value, ClassLabel  # Option B commented out
from transformers import AutoTokenizer, PreTrainedTokenizer

"""
genomics_csv_dataloader.py

A custom data loader for genomic sequence classification using CSV-formatted data.

The CSV should have at least two columns: 'sequence' and 'label'.
Labels can be integers or strings; if strings, map them to integers manually.

Option A (GenomicsCSVDataset): Represents the approach used specifically in the HyenaDNA example, customized for genomic datasets.
# Option B (load_huggingface_csv_dataset): Represents the general-purpose Hugging Face loader for CSV files.
"""

# Option A: GenomicsCSVDataset (Hyena example)
class GenomicsCSVDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, max_length: int, label2id: dict = None):
        """
        file_path: Path to CSV file containing 'sequence' and 'label' columns.
        tokenizer: Hugging Face tokenizer.
        max_length: max token length for sequences.
        label2id: Optional dict mapping string labels to integer ids.
        """
        df = pd.read_csv(file_path)
        if label2id:
            df['label'] = df['case_control'].map(label2id)
        else:
            df['label'] = df['case_control']
        self.sequences = df['sequence'].tolist()
        self.labels = df['label'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        enc = self.tokenizer(seq,
                             add_special_tokens=False,
                             max_length=self.max_length,
                             padding='max_length',
                             truncation=True,
                             return_attention_mask=True,
                             return_tensors='pt')
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item['labels'] = torch.tensor(label, dtype=torch.long)
        return item

# Option B: load_huggingface_csv_dataset (Hugging Face example) - commented out for large sequence support
# def load_huggingface_csv_dataset(file_path: str, tokenizer: PreTrainedTokenizer, max_length: int):
#     """
#     Alternative: load using Hugging Face datasets.
#     """
#     # detect unique labels
#     df = pd.read_csv(file_path)
#     label_list = sorted(df['label'].unique().tolist())
#     features = Features({'sequence': Value('string'),
#                          'label': ClassLabel(names=label_list)})
#     dataset = load_dataset('csv',
#                            data_files=[file_path],
#                            features=features,
#                            split='train')
#     def tokenize_fn(example):
#         enc = tokenizer(example['sequence'],
#                          add_special_tokens=False,
#                          padding='max_length',
#                          truncation=True,
#                          max_length=max_length)
#         enc['labels'] = example['label']
#         return enc
#     dataset = dataset.map(tokenize_fn, batched=True, batch_size=1)  # safer for long sequences
#     dataset.set_format(type='torch', columns=['input_ids','attention_mask','labels'])
#     return dataset

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(
    "/sc/arion/projects/ipm/aysegul/models/hyena-dna/hyenadna-large-1m-seqlen",
    trust_remote_code=True,
    local_files_only=True)
    max_len = 1_000_000
    train_file = os.getenv('TRAIN_CSV', '/sc/arion/projects/mscic1/PRS-LLM/data/clumped/ayub/ayub_clumped_samples_sequences_with_labels_withSNP.csv')
    print("Using Option A: GenomicsCSVDataset")
    label2id = {'control':0, 'carrier':1}
    dataset_a = GenomicsCSVDataset(train_file, tokenizer, max_len, label2id)
    dataloader = DataLoader(dataset_a, batch_size=1, shuffle=True)
    for batch in dataloader:
        print(batch['input_ids'].shape, batch['labels'])
        break

    # Option B example commented out for safety with long sequences
    # print("\nUsing Option B: load_huggingface_csv_dataset")
    # dataset_b = load_huggingface_csv_dataset(train_file, tokenizer, max_len)
    # print(dataset_b[0]['input_ids'].shape, dataset_b[0]['labels'])


    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        if pd.isna(label):
            raise ValueError(f"Label at index {idx} is NaN!")
        enc = self.tokenizer(seq,
                             add_special_tokens=False,
                             max_length=self.max_length,
                             padding='max_length',
                             truncation=True,
                             return_attention_mask=True,
                             return_tensors='pt')
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item['labels'] = torch.tensor(int(label), dtype=torch.long)
        return item


class GenomicsCSVDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, max_length: int, label2id: dict = None):
        """
        file_path: Path to CSV file containing 'sequence' and 'label' columns.
        tokenizer: Hugging Face tokenizer.
        max_length: max token length for sequences.
        label2id: Optional dict mapping string labels to integer ids.
        """
        df = pd.read_csv(file_path)
        if label2id:
            df['label'] = df['case_control'].map(label2id)
        else:
            df['label'] = df['case_control']
        self.sequences = df['sequence'].tolist()
        self.labels = df['label'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        enc = self.tokenizer(seq,
                             add_special_tokens=False,
                             max_length=self.max_length,
                             padding='max_length',
                             truncation=True,
                             return_attention_mask=True,
                             return_tensors='pt')
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item['labels'] = torch.tensor(label, dtype=torch.long)
        return item






