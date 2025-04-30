import os
from datasets import load_dataset, Features, Value, ClassLabel, Dataset
from typing import Optional, List

def load_huggingface_csv_dataset(
    file_path: str,
    label_column: str = 'case_control',
    sequence_column: str = 'sequence',
    label_list: Optional[List[str]] = None,
    cache_dir: Optional[str] = None,
    save_to_disk_dir: Optional[str] = None,
) -> Dataset:
    """
    Load a Hugging Face Dataset from a CSV file containing genomic sequences and labels.
    Parameters:
    - file_path (str): Path to the input CSV.
    - label_column (str): Name of the column containing labels.
    - sequence_column (str): Name of the column containing sequences.
    - label_list (List[str], optional): Predefined list of label names. If None, inferred from data.
    - cache_dir (str, optional): Directory for Hugging Face cache.
    - save_to_disk_dir (str, optional): If provided, saves the resulting dataset to disk.
    Returns:
    - Dataset: Hugging Face Dataset with proper features.
    """
    print(f"Loading dataset from: {file_path}")
    # Infer label list if not provided
    if label_list is None:
        temp_dataset = load_dataset('csv', data_files=file_path, split='train')
        label_list = sorted(set(temp_dataset[label_column]))
        del temp_dataset  # Free memory
    features = Features({
        sequence_column: Value('string'),
        label_column: Value('int64')
    })
    dataset = load_dataset(
        'csv',
        data_files=file_path,
        features=features,
        split='train',
        cache_dir=cache_dir
    )
    print(f"Loaded dataset with {len(dataset)} examples.")
    print("Example row:", dataset[0])
    print("Label classes:", label_list)
    if save_to_disk_dir:
        print(f"Saving dataset to disk at: {save_to_disk_dir}")
        dataset.save_to_disk(save_to_disk_dir)
    return dataset


# ------------------------ RUN BELOW ------------------------

if __name__ == "__main__":
    # Provide your CSV file path here
    file_path = "/sc/arion/projects/mscic1/PRS-LLM/data/clumped/ayub/ayub_clumped_samples_sequences_with_labels_withSNP.csv"
    # Optional: Set caching and save paths
    cache_dir = "/sc/arion/projects/mscic1/PRS-LLM/hf_cache"
#    save_dir = "/sc/arion/projects/mscic1/PRS-LLM/datasets/ayub_dataset"
    # Load the dataset
    dataset = load_huggingface_csv_dataset(
        file_path=file_path,
        cache_dir=cache_dir,
 #       save_to_disk_dir=save_dir
    )




