import os
from datasets import load_dataset, Features, Value, Dataset
from typing import Optional

def load_huggingface_csv_dataset(
    file_path: str,
    cache_dir: Optional[str] = None,
    save_to_disk_dir: Optional[str] = None,
) -> Dataset:
    """
    Load a Hugging Face Dataset from a CSV file containing genomic sequences and labels.
    Assumes CSV columns:
    - Sample (str)
    - sequence (str)
    - case_control (int)
    - dataset (str)
    Parameters:
    - file_path (str): Path to the input CSV.
    - cache_dir (str, optional): Directory for Hugging Face cache.
    - save_to_disk_dir (str, optional): If provided, saves the resulting dataset to disk.
    Returns:
    - Dataset: Hugging Face Dataset with proper features.
    """
    print(f"Loading dataset from: {file_path}")
    features = Features({
        "sample": Value("string"),
        "sequence": Value("string"),
        "labels": Value("int64"),
        "dataset": Value("string")
    })
    dataset = load_dataset(
        "csv",
        data_files=file_path,
        features=features,
        split="train",
        cache_dir=cache_dir
    )
    print(f"Loaded dataset with {len(dataset)} examples.")
    print("Example row:", dataset[0])
    print("Label classes:", sorted(set(dataset['labels'])))
    if save_to_disk_dir:
        print(f"Saving dataset to disk at: {save_to_disk_dir}")
        dataset.save_to_disk(save_to_disk_dir)
    return dataset


# ------------------------ RUN BELOW ------------------------

if __name__ == "__main__":
    file_path = "/sc/arion/projects/mscic1/PRS-LLM/data/clumped/ayub_clumped_samples_sequences_with_labels_withSNP.csv"
    cache_dir = "/sc/arion/projects/mscic1/PRS-LLM/hf_cache"
    # save_dir = "/sc/arion/projects/mscic1/PRS-LLM/datasets/ayub_dataset"
    dataset = load_huggingface_csv_dataset(
        file_path=file_path,
        cache_dir=cache_dir,
        # save_to_disk_dir=save_dir
    )
