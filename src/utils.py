import json
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from transformers import BertForMaskedLM, BertForSequenceClassification, \
    BertTokenizer, BertConfig

ORIG_MISCLASSIFIED = 3
ATTACK_SUCCESSFUL = 4


@dataclass
class Feature(object):
    """Stores attack statistics for a sample.

    TODO Complete missing attribute descriptions.
    Attributes:
        label: Assigned class.
        seq: String sequence.
        final_adverse:
        query: Total number of queries performed.
        change:
        success:
        sim:
        changes: Changes to the source sentence. Each entry contains
        [sub_word_start, substituted_word, target_word].
    """

    def __init__(self, seq_a, label):
        self.label: int = label
        self.seq: str = seq_a
        self.final_adverse: str = seq_a
        self.query: int = 0
        self.change: int = 0
        self.success: int = 0
        self.sim: float = 0.0
        self.changes: list = []


@dataclass
class BigFeature(object):
    """Stores attack statistics (with text and results) for a sample.

    TODO Complete missing attribute descriptions.
    Attributes:
        label: Assigned class.
        seq: String sequence.
        orig_probs: Probabilities for the original sequence.
        adv_texts: List of all adversarial texts.
        adv_probs: List of all predictions corresponding to texts.
        query: Total number of queries performed.
        change: Total number of words changed.
        success: 
        sim: Similarity to the original sequence.
        changes: Changes to the source sentence. Each entry contains
        [sub_word_start, substituted_word, target_word].
    """

    def __init__(self, seq_a, label):
        self.label: int = label
        self.seq: str = seq_a
        self.orig_probs: List[float] = []
        self.adv_texts: List[Tuple[str, str, str]] = []
        self.adv_probs: List[List[float]] = []
        self.query: int = 0
        self.change: int = 0
        self.success: int = 0
        self.sim: float = 0.0
        self.changes: list = []


def load_similarity_embed(embed_path: str, sim_path: str) -> (np.ndarray, dict,
                                                              dict):
    """Loads similarity embedding vectors."""
    idx2word = {}
    word2idx = {}

    with open(embed_path, 'r', encoding='utf-8') as input_file:
        for line in input_file:
            word = line.split()[0]
            if word not in idx2word:
                idx2word[len(idx2word)] = word
                word2idx[word] = len(idx2word) - 1

    cos_sim = np.load(sim_path)
    return cos_sim, word2idx, idx2word


def load_dataset(data_path: str) -> List[Tuple[str, int]]:
    """Loads the dataset.

    Args:
        data_path: Path to TSV file.

    Returns:
        A list of sequence(str), label(int) pairs.
    """
    # Open file and skip the first line
    lines = open(data_path, 'r', encoding='utf-8').readlines()[1:]
    features = []
    for line in lines:
        split = line.strip('\n').split('\t')
        seq, label = split[0], int(split[-1])
        features.append((seq, label))
    return features


def load_models(args: dict) -> (BertForMaskedLM, BertForSequenceClassification,
                                BertTokenizer):
    """Loads the MLM and classification models along with the tokenizer."""
    # Load tokenizers
    tokenizer_tgt = BertTokenizer.from_pretrained(args["tgt_path"],
                                                  do_lower_case=True)

    # Load MLM model
    config_atk = BertConfig.from_pretrained(args["mlm_path"])
    mlm_model = BertForMaskedLM.from_pretrained(args["mlm_path"],
                                                config=config_atk)
    mlm_model.to('cuda')

    # Load target model
    config_tgt = BertConfig.from_pretrained(args["tgt_path"],
                                            num_labels=args["num_label"])
    tgt_model = BertForSequenceClassification.from_pretrained(args["tgt_path"],
                                                              config=config_tgt)
    tgt_model.to('cuda')

    return mlm_model, tgt_model, tokenizer_tgt


def dump_features(features: List[Feature], out_path: str):
    """Writes experiment output to file."""
    outputs = []
    for feature in features:
        outputs.append({
            'label': feature.label,
            'success': feature.success,
            'change': feature.change,
            'num_word': len(feature.seq.split(' ')),
            'query': feature.query,
            'changes': feature.changes,
            'seq_a': feature.seq,
            'adv': feature.final_adverse,
        })

    json.dump(outputs, open(out_path, 'w'), indent=2)
