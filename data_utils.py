import logging
from datasets import load_dataset
from transformers import AutoTokenizer
import random

def load_dataset_streaming(dataset_name, subset, split, max_samples):
    logging.info(f"Loading dataset {dataset_name} ({subset})...")
    if dataset_name == "wikitext":
        dataset = load_dataset(dataset_name, subset, split=split)
        logging.info(f"Dataset loaded successfully! Total samples: {len(dataset)}")
    elif dataset_name == "c4":
        dataset = load_dataset("allenai/c4",  data_files="en/c4-train.0000*-of-01024.json.gz", split=split)
        if max_samples is not None:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        logging.info(f"Dataset loaded successfully! Using {len(dataset)} samples from C4 dataset.")
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return dataset

def get_wikitext2(nsamples, seed, seqlen, model_path):
    _ = load_dataset_streaming("wikitext", "wikitext-2-raw-v1", "train", nsamples)
    test_dataset = load_dataset_streaming("wikitext", "wikitext-2-raw-v1", "test", nsamples)
    return test_dataset

def get_c4(nsamples, seed, seqlen, model_path, full_dataset=False):
    # _ = load_dataset_streaming("c4", "en", "train", nsamples)
    val_dataset = load_dataset_streaming("c4", "en", "train", nsamples)
    return val_dataset 