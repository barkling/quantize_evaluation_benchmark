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
        dataset = load_dataset("c4", "en", split=split)
        if max_samples is not None:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        logging.info(f"Dataset loaded successfully! Using {len(dataset)} samples from C4 dataset.")
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return dataset

def get_wikitext2(nsamples, seed, seqlen, model_path):
    dataset = load_dataset_streaming("wikitext", "wikitext-2-raw-v1", "train", nsamples)
    test_dataset = load_dataset_streaming("wikitext", "wikitext-2-raw-v1", "test", nsamples)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    trainenc = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
    testenc = tokenizer("\n\n".join(test_dataset["text"]), return_tensors="pt")
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    if len(trainloader) == 0:
        raise ValueError(f"No training samples could be created with seqlen={seqlen}. "
                         f"Please use a smaller sequence length or provide more data.")
    return trainloader, test_dataset

def get_c4(nsamples, seed, seqlen, model_path, full_dataset=False):
    dataset = load_dataset_streaming("c4", "en", "train", nsamples)
    val_dataset = load_dataset_streaming("c4", "en", "validation", nsamples)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    random.seed(seed)
    trainloader = []
    if full_dataset:
        logging.info("Using full C4 dataset for evaluation")
        for example in dataset:
            text = example["text"]
            if not text.strip():
                continue
            trainenc = tokenizer(text, return_tensors="pt")
            if trainenc.input_ids.shape[1] < seqlen:
                continue
            for i in range(0, trainenc.input_ids.shape[1] - seqlen, seqlen):
                j = i + seqlen
                inp = trainenc.input_ids[:, i:j]
                tar = inp.clone()
                tar[:, :-1] = -100
                trainloader.append((inp, tar))
                if len(trainloader) >= nsamples:
                    break
            if len(trainloader) >= nsamples:
                break
    else:
        logging.info("Using random C4 fragments for evaluation")
        for example in dataset:
            text = example["text"]
            if not text.strip():
                continue
            trainenc = tokenizer(text, return_tensors="pt")
            if trainenc.input_ids.shape[1] < seqlen:
                continue
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
            if len(trainloader) >= nsamples:
                break
    if len(trainloader) == 0:
        raise ValueError(f"No training samples could be created with seqlen={seqlen}. "
                         f"Please use a smaller sequence length or provide more data.")
    return trainloader, val_dataset 