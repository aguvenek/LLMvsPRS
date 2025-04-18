"""
genomics_csv_dataloader.py

A custom data loader for genomic sequence classification using CSV-formatted data.

The CSV should have at least two columns: 'sequence' and 'label'.
Labels can be integers or strings; if strings, map them to integers manually.
"""

import os
import pandas as pd
torching_import = "import torch"
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Features, Value, ClassLabel
from transformers import PreTrainedTokenizer

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
            df['label'] = df['label'].map(label2id)
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


def load_huggingface_csv_dataset(file_path: str, tokenizer: PreTrainedTokenizer, max_length: int):
    """
    Alternative: load using Hugging Face datasets.
    """
    # detect unique labels
    df = pd.read_csv(file_path)
    label_list = sorted(df['label'].unique().tolist())
    features = Features({'sequence': Value('string'),
                         'label': ClassLabel(names=label_list)})
    dataset = load_dataset('csv',
                           data_files=[file_path],
                           features=features,
                           split='train')
    def tokenize_fn(example):
        enc = tokenizer(example['sequence'],
                         add_special_tokens=False,
                         padding='max_length',
                         truncation=True,
                         max_length=max_length)
        enc['labels'] = example['label']
        return enc
    dataset = dataset.map(tokenize_fn, batched=False)
    dataset.set_format(type='torch', columns=['input_ids','attention_mask','labels'])
    return dataset


if __name__ == '__main__':
    from transformers import AutoTokenizer
    import torch
    # Example usage
    tokenizer = AutoTokenizer.from_pretrained('LongSafari/hyenadna-medium-160k-seqlen-hf', trust_remote_code=True)
    max_len = 160000
    train_file = os.getenv('TRAIN_CSV', 'sc/arion/projects/mscic1/PRS-LLM/data/clumped/ayub/ayub_clumped_samples_sequences_with_labels_withSNP.csv')
    # use string labels mapping:
    label2id = {'control':0, 'carrier':1}
    dataset = GenomicsCSVDataset(train_file, tokenizer, max_len, label2id)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for batch in dataloader:
        print(batch['input_ids'].shape, batch['labels'])
        break
