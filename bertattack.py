# Adapted from https://github.com/LinyangLee/BERT-Attack
#
# Original information as follows:
# -*- coding: utf-8 -*-
# @Time    : 2020/6/10
# @Author  : Linyang Li
# @Email   : linyangli19@fudan.edu.cn
# @File    : attack.py


import warnings
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
import copy
import argparse
from src.evaluate import evaluate
from src.utils import load_similarity_embed, load_dataset, Feature,\
    dump_features, load_models
from transformers import BertForMaskedLM, BertForSequenceClassification,\
    BertTokenizer
from typing import Optional, List, Tuple

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)

FILTER_WORDS = [
    'a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against',
    'ain', 'all', 'almost', 'alone', 'along', 'already', 'also', 'although',
    'am', 'among', 'amongst', 'an', 'and', 'another', 'any', 'anyhow', 'anyone',
    'anything', 'anyway', 'anywhere', 'are', 'aren', "aren't", 'around', 'as',
    'at', 'back', 'been', 'before', 'beforehand', 'behind', 'being', 'below',
    'beside', 'besides', 'between', 'beyond', 'both', 'but', 'by', 'can',
    'cannot', 'could', 'couldn', "couldn't", 'd', 'didn', "didn't", 'doesn',
    "doesn't", 'don', "don't", 'down', 'due', 'during', 'either', 'else',
    'elsewhere', 'empty', 'enough', 'even', 'ever', 'everyone', 'everything',
    'everywhere', 'except', 'first', 'for', 'former', 'formerly', 'from',
    'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'he', 'hence',
    'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers',
    'herself', 'him', 'himself', 'his', 'how', 'however', 'hundred', 'i', 'if',
    'in', 'indeed', 'into', 'is', 'isn', "isn't", 'it', "it's", 'its', 'itself',
    'just', 'latter', 'latterly', 'least', 'll', 'may', 'me', 'meanwhile',
    'mightn', "mightn't", 'mine', 'more', 'moreover', 'most', 'mostly', 'must',
    'mustn', "mustn't", 'my', 'myself', 'namely', 'needn', "needn't", 'neither',
    'never', 'nevertheless', 'next', 'no', 'nobody', 'none', 'noone', 'nor',
    'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'on', 'once', 'one',
    'only', 'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours',
    'ourselves', 'out', 'over', 'per', 'please', 's', 'same', 'shan', "shan't",
    'she', "she's", "should've", 'shouldn', "shouldn't", 'somehow', 'something',
    'sometime', 'somewhere', 'such', 't', 'than', 'that', "that'll", 'the',
    'their', 'theirs', 'them', 'themselves', 'then', 'thence', 'there',
    'thereafter', 'thereby', 'therefore', 'therein', 'thereupon', 'these',
    'they', 'this', 'those', 'through', 'throughout', 'thru', 'thus', 'to',
    'too', 'toward', 'towards', 'under', 'unless', 'until', 'up', 'upon',
    'used', 've', 'was', 'wasn', "wasn't", 'we', 'were', 'weren', "weren't",
    'what', 'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter',
    'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether',
    'which', 'while', 'whither', 'who', 'whoever', 'whole', 'whom', 'whose',
    'why', 'with', 'within', 'without', 'won', "won't", 'would', 'wouldn',
    "wouldn't", 'y', 'yet', 'you', "you'd", "you'll", "you're", "you've",
    'your', 'yours', 'yourself', 'yourselves',
]
FILTER_WORDS = set(FILTER_WORDS)


def tokenize(seq, tokenizer) -> (List[str], List[str], List[Tuple[int, int]]):
    seq = seq.replace('\n', '').lower()
    words = seq.split(' ')

    sub_words = []
    keys = []
    index = 0
    for word in words:
        sub = tokenizer.tokenize(word)
        sub_words += sub
        keys.append((index, index + len(sub)))
        index += len(sub)

    return words, sub_words, keys


def mask_sentence(original: List[str]) -> List[str]:
    """Generates all distinct sentences with a single token masked.

    Args:
        original: The tokenized source sentence.
    Returns:
        A list of sentences with one token masked.
    """
    split_texts = [original[0:i] + ['[UNK]'] + original[i + 1:] for i in
                   range(len(original) - 1)]
    texts = [' '.join(words) for words in split_texts]
    return texts


def generate_importance_scores(source_sent: List[str],
                               tgt_model: BertForSequenceClassification,
                               orig_prob, orig_label, orig_probs,
                               tokenizer: BertTokenizer, batch_size,
                               max_length):
    # Create a list of masked sentences
    texts = mask_sentence(source_sent)
    all_input_ids = []

    for text in texts:
        inputs = tokenizer.encode_plus(
            text,
            None, add_special_tokens=True, max_length=max_length,
            truncation='longest_first'
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        padding_length = max_length - len(input_ids)
        input_ids = input_ids + (padding_length * [0])
        all_input_ids.append(input_ids)
    seqs = torch.tensor(all_input_ids, dtype=torch.long)
    seqs = seqs.to('cuda')

    eval_data = TensorDataset(seqs)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)
    leave_1_probs = []
    for batch in eval_dataloader:
        masked_input, = batch
        leave_1_prob_batch = tgt_model(masked_input)[0]  # B num-label
        leave_1_probs.append(leave_1_prob_batch)

    leave_1_probs = torch.cat(leave_1_probs, dim=0)  # words, num-label
    leave_1_probs = torch.softmax(leave_1_probs, -1)  #
    leave_1_probs_argmax = torch.argmax(leave_1_probs, dim=-1)
    # noinspection PyUnresolvedReferences
    import_scores = (
        orig_prob
        - leave_1_probs[:, orig_label]
        +
        (leave_1_probs_argmax != orig_label).float()
        * (leave_1_probs.max(dim=-1)[0] - torch.index_select(orig_probs, 0, leave_1_probs_argmax))
    ).data.cpu().numpy()

    return import_scores


def get_substitutes(substitutes, tokenizer, mlm_model, use_bpe, substitutes_score=None, threshold=3.0):
    # substitutes L,k
    # from this matrix to recover a word
    words = []
    sub_len, k = substitutes.size()  # sub-len, k

    if sub_len == 0:
        return words
        
    elif sub_len == 1:
        for (i, j) in zip(substitutes[0], substitutes_score[0]):
            if threshold != 0 and j < threshold:
                break
            words.append(tokenizer._convert_id_to_token(int(i)))
    else:
        if use_bpe == 1:
            words = get_bpe_substitutes(substitutes, tokenizer, mlm_model)
        else:
            return words
    #
    # print(words)
    return words


def get_bpe_substitutes(substitutes, tokenizer, mlm_model):
    # substitutes L, k

    substitutes = substitutes[0:12, 0:4]  # maximum BPE candidates

    # find all possible candidates 

    all_substitutes = []
    for i in range(substitutes.size(0)):
        if len(all_substitutes) == 0:
            lev_i = substitutes[i]
            all_substitutes = [[int(c)] for c in lev_i]
        else:
            lev_i = []
            for all_sub in all_substitutes:
                for j in substitutes[i]:
                    lev_i.append(all_sub + [int(j)])
            all_substitutes = lev_i

    # all substitutes  list of list of token-id (all candidates)
    c_loss = nn.CrossEntropyLoss(reduction='none')
    # word_list = []
    # all_substitutes = all_substitutes[:24]
    all_substitutes = torch.tensor(all_substitutes) # [ N, L ]
    all_substitutes = all_substitutes[:24].to('cuda')
    # print(substitutes.size(), all_substitutes.size())
    N, L = all_substitutes.size()
    word_predictions = mlm_model(all_substitutes)[0]  # N L vocab-size
    ppl = c_loss(word_predictions.view(N*L, -1), all_substitutes.view(-1))  # [ N*L ]
    ppl = torch.exp(torch.mean(ppl.view(N, L), dim=-1)) # N  
    _, word_list = torch.sort(ppl)
    word_list = [all_substitutes[i] for i in word_list]
    final_words = []
    for word in word_list:
        tokens = [tokenizer._convert_id_to_token(int(i)) for i in word]
        text = tokenizer.convert_tokens_to_string(tokens)
        final_words.append(text)
    return final_words


def attack(feature: Feature, tgt_model: BertForSequenceClassification,
           mlm_model: BertForMaskedLM, tokenizer: BertTokenizer, k: int,
           batch_size: int, max_length: int = 512,
           cos_mat: Optional[np.ndarray] = None, w2i: Optional[dict] = None,
           i2w: Optional[dict] = None, use_bpe: int = 1,
           threshold_pred_score: float = 0.3
):
    """Performs all substitution attacks on the BERT model."""
    # MLM-process
    if i2w is None:
        i2w = {}
    if w2i is None:
        w2i = {}
    words, sub_words, keys = tokenize(feature.seq, tokenizer)

    # Prepare feature for prediction
    inputs = tokenizer.encode_plus(
        feature.seq, None, add_special_tokens=True, max_length=max_length,
        truncation='longest_first'
    )
    input_ids, token_type_ids = torch.tensor(inputs["input_ids"]), torch.tensor(
        inputs["token_type_ids"])
    attention_mask = torch.tensor([1] * len(input_ids))

    # Perform prediction on original sample
    orig_predictions = tgt_model(
        input_ids.unsqueeze(0).to('cuda'),
        attention_mask.unsqueeze(0).to('cuda'),
        token_type_ids.unsqueeze(0).to('cuda')
    )[0].squeeze()
    orig_predictions = torch.softmax(orig_predictions, -1)
    orig_label = torch.argmax(orig_predictions)
    current_prob = orig_predictions.max()

    if orig_label != feature.label:
        feature.success = 3
        return feature

    sub_words = ['[CLS]'] + sub_words[:max_length - 2] + ['[SEP]']
    input_ids_ = torch.tensor([tokenizer.convert_tokens_to_ids(sub_words)])
    word_predictions = mlm_model(input_ids_.to('cuda'))[0].squeeze()  # seq-len(sub) vocab
    word_pred_scores_all, word_predictions = torch.topk(word_predictions, k, -1)  # seq-len k

    word_predictions = word_predictions[1:len(sub_words) + 1, :]
    word_pred_scores_all = word_pred_scores_all[1:len(sub_words) + 1, :]

    important_scores = generate_importance_scores(words, tgt_model, current_prob, orig_label, orig_predictions,
                                                  tokenizer, batch_size, max_length)
    feature.query += int(len(words))
    list_of_index = sorted(enumerate(important_scores), key=lambda x: x[1], reverse=True)
    # print(list_of_index)
    final_words = copy.deepcopy(words)

    for top_index in list_of_index:
        if feature.change > int(0.4 * (len(words))):
            feature.success = 1  # exceed
            return feature

        tgt_word = words[top_index[0]]
        if tgt_word in FILTER_WORDS:
            continue
        if keys[top_index[0]][0] > max_length - 2:
            continue

        substitutes = word_predictions[keys[top_index[0]][0]:keys[top_index[0]][1]]  # L, k
        word_pred_scores = word_pred_scores_all[keys[top_index[0]][0]:keys[top_index[0]][1]]

        substitutes = get_substitutes(substitutes, tokenizer, mlm_model, use_bpe, word_pred_scores,
                                      threshold_pred_score)

        most_gap = 0.0
        candidate = None

        for substitute_ in substitutes:
            substitute = substitute_

            if substitute == tgt_word:
                continue  # filter out original word
            if '##' in substitute:
                continue  # filter out sub-word

            if substitute in FILTER_WORDS:
                continue
            if substitute in w2i and tgt_word in w2i:
                if cos_mat[w2i[substitute]][w2i[tgt_word]] < 0.4:
                    continue
            temp_replace = final_words
            temp_replace[top_index[0]] = substitute
            temp_text = tokenizer.convert_tokens_to_string(temp_replace)
            inputs = tokenizer.encode_plus(temp_text, None, add_special_tokens=True, max_length=max_length, truncation='longest_first')
            input_ids = torch.tensor(inputs["input_ids"]).unsqueeze(0).to('cuda')
            seq_len = input_ids.size(1)
            temp_prob = tgt_model(input_ids)[0].squeeze()
            feature.query += 1
            temp_prob = torch.softmax(temp_prob, -1)
            temp_label = torch.argmax(temp_prob)

            if temp_label != orig_label:
                feature.change += 1
                final_words[top_index[0]] = substitute
                feature.changes.append([keys[top_index[0]][0], substitute, tgt_word])
                feature.final_adverse = temp_text
                feature.success = 4
                return feature
            else:

                label_prob = temp_prob[orig_label]
                gap = current_prob - label_prob
                if gap > most_gap:
                    most_gap = gap
                    candidate = substitute

        if most_gap > 0:
            feature.change += 1
            feature.changes.append([keys[top_index[0]][0], candidate, tgt_word])
            current_prob = current_prob - most_gap
            final_words[top_index[0]] = candidate

    feature.final_adverse = (tokenizer.convert_tokens_to_string(final_words))
    feature.success = 2
    return feature


def main(args: dict):
    print('Starting process...')
    # Load required models, tokenizers, and dataset
    model_mlm, model_target, tokenizer_target = load_models(args)
    samples = load_dataset(args["data_path"])

    if args["use_sim_mat"] == 1:
        print('\tLoading similarity embeddings...')
        cos_mat, w2i, i2w = load_similarity_embed(
            'data_defense/counter-fitted-vectors.txt',
            'data_defense/cos_sim_counter_fitting.npy',
        )
    else:        
        cos_mat, w2i, i2w = None, {}, {}

    print("\tPerforming attacks...")
    features_output = []
    with torch.no_grad():
        print()
        n = args["end"] - args["start"]
        for index, feature in enumerate(samples[args["start"]:args["end"]]):
            # Convert sample to object with embedded metrics
            feature = Feature(*feature)

            # Perform attacks on the sample
            print('\r\t\t[{:d} / {:d}] '.format(index, n), end='')
            feature = attack(
                feature, model_target, model_mlm, tokenizer_target, args["k"],
                batch_size=32, max_length=512, cos_mat=cos_mat, w2i=w2i,
                i2w=i2w, use_bpe=args["use_bpe"],
                threshold_pred_score=args["threshold_pred_score"]
            )

            if feature.success > 2:
                print('Successful', end='')
            else:
                print('Failed', end='')
            features_output.append(feature)

    # Evaluate and save
    print("\r\tEvaluating performance")
    evaluate(features_output)

    print("\r\tSaving files")
    dump_features(features_output, args["output_dir"])

    print("Completed")


def parse_args() -> dict:
    """Load and parse arguments.

    The parser accepts the following arguments:
        data_path: Path to the TSV dataset file.
        mlm_path: Path to the directory containing the MLM model.
        tgt_path: Path to the directory containing the classifier.
        output_dir: Path to experiment results. Must be a directory.
        use_sim_mat: Flag to set if cosine similarity is used to filter antonyms.
        start: Starting step. Usable for multi-threaded processing.
        start: Ending step. Usable for multi-threaded processing.
        num_label: Number of labels.
        use_bpe: TODO Read about BPE and fill in
        k: Number of words to be tested for replacement.
        threshold_pred_score: Positive prediction threshold.
    """
    # Parse all arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path to the TSV dataset file.")
    parser.add_argument("--mlm_path", type=str, help="Path to the directory containing the MLM model.")
    parser.add_argument("--tgt_path", type=str, help="Path to the directory containing the classifier.")
    parser.add_argument("--output_dir", type=str, help="Path to experiment results. Must be a directory.")
    parser.add_argument("--use_sim_mat", type=int, help='Flag to set if cosine similarity is used to filter antonyms.')
    parser.add_argument("--start", type=int, help="Starting step. Usable for multi-threaded processing.")
    parser.add_argument("--end", type=int, help="Ending step. Usable for multi-threaded processing.")
    parser.add_argument("--num_label", type=int, help="Number of labels")
    parser.add_argument("--use_bpe", type=int, help="")
    parser.add_argument("--k", type=int, help="Number of words to be tested for replacement.")
    parser.add_argument("--threshold_pred_score", type=float, help="Positive prediction threshold.")
    args = parser.parse_args()

    # Return namespace
    return vars(args)


if __name__ == '__main__':
    # Parse arguments
    args = parse_args()
    main(args)
