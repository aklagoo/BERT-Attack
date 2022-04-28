import copy
from typing import List, Tuple, Optional
import numpy as np
import torch
from torch import nn as nn
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
from transformers import BertForSequenceClassification, BertTokenizer, \
    BertForMaskedLM
from src.utils import Feature, ORIG_MISCLASSIFIED, BigFeature, ATTACK_SUCCESSFUL

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


def tokenize(seq: str, tokenizer: BertTokenizer) -> (List[str], List[str],
                                                     List[Tuple[int, int]]):
    """Tokenizes text into words and BERT sub-words with mapping.

    The function splits each word in the sentence into sub-words based on the
    BERT tokenizer provided. Three lists are maintained: words, sub-words, and
    keys. Keys maps words to sub-word indices.

    Example:
        Assume that the 3rd word 'abc' is split into 2 sub-words 'ab' and 'c'.

        words[3] = 'abc'
        keys[3] = (9, 11)
        sub_words[9:11] = ['ab', 'c']
    """
    # Convert to lowercase and split
    seq = seq.replace('\n', '').lower()
    words = seq.split(' ')

    # Tokens for BERT
    sub_words = []

    # Word-to-Sub-word index mapping
    keys = []

    index = 0
    for word in words:
        # Tokenize words
        sub = tokenizer.tokenize(word)
        sub_words += sub
        keys.append((index, index + len(sub)))
        index += len(sub)

    return words, sub_words, keys


def mask_sentence(original: List[str]) -> List[str]:
    """Generates all distinct sentences with a single token masked."""
    # Replace all words except the last one, which is the period.
    split_texts = [original[0:i] + ['[UNK]'] + original[i + 1:] for i in
                   range(len(original) - 1)]
    texts = [' '.join(words) for words in split_texts]
    return texts


def generate_importance_scores(source_sent: List[str],
                               tgt_model: BertForSequenceClassification,
                               orig_prob, orig_label, orig_probs,
                               tokenizer: BertTokenizer, batch_size: int,
                               max_length: int):
    """
    Generates importance scores by masking words in the sentences.

    Args:
        source_sent: Source sentence tokenized as whole words, i.e., no
         sub-words are used.
        tgt_model: The classification model used.
        orig_prob: Probability of the original label. Single value.
        orig_label: The original label. Single index.
        orig_probs: Probabilities for all labels, with a shape [num_labels,].
        tokenizer: The BERT tokenizer.
        batch_size: The batch size for generating predictions.
        max_length: The max length for tokenization.

    Returns:
        A 1-D tensor of importance scores. One for each word, excluding the
        final period ('.').
    """
    # Create a list of masked sentences. Every word except the final '.' token
    # is replaced. This results in the length:
    #     `num_masked = num_words - 1`.
    texts = mask_sentence(source_sent)
    all_input_ids = []

    # Convert batch of masked sentences to tensor
    for text in texts:
        # Tokenize text
        inputs = tokenizer.encode_plus(
            text,
            None, add_special_tokens=True, max_length=max_length,
            truncation='longest_first'
        )
        input_ids = inputs["input_ids"]

        # Pad data
        padding_length = max_length - len(input_ids)
        input_ids = input_ids + (padding_length * [0])

        all_input_ids.append(input_ids)
    seqs = torch.tensor(all_input_ids, dtype=torch.long)
    seqs = seqs.to('cuda')  # num_masked x max_length

    # Create dataset
    eval_data = TensorDataset(seqs)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler,
                                 batch_size=batch_size)
    probs = []

    for batch in eval_dataloader:
        masked_input, = batch
        probs_batch = tgt_model(masked_input)[0]
        probs.append(probs_batch)

    # Stack all batches to get a single array of results. Then, calculate the
    # predicted labels.
    probs = torch.cat(probs, dim=0)  # num_masked x num_labels
    probs = torch.softmax(probs, -1)
    probs_argmax = torch.argmax(probs, dim=-1)

    # Calculate one importance score for each word.
    import_scores = orig_prob - probs[:, orig_label] + (probs_argmax !=
                                                        orig_label).float() * (
            probs.max(dim=-1)[0] - torch.index_select(orig_probs, 0,
                                                      probs_argmax))
    import_scores = import_scores.data.cpu().numpy()  # num_masked

    print("Original prob" + str(orig_prob))

    return import_scores


def get_substitutes(substitutes, tokenizer, mlm_model, use_bpe,
                    substitutes_score=None, threshold=3.0):
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
    all_substitutes = torch.tensor(all_substitutes)  # [ N, L ]
    all_substitutes = all_substitutes[:24].to('cuda')
    # print(substitutes.size(), all_substitutes.size())
    N, L = all_substitutes.size()
    word_predictions = mlm_model(all_substitutes)[0]  # N L vocab-size
    ppl = c_loss(word_predictions.view(N * L, -1),
                 all_substitutes.view(-1))  # [ N*L ]
    ppl = torch.exp(torch.mean(ppl.view(N, L), dim=-1))  # N
    _, word_list = torch.sort(ppl)
    word_list = [all_substitutes[i] for i in word_list]
    final_words = []
    for word in word_list:
        tokens = [tokenizer._convert_id_to_token(int(i)) for i in word]
        text = tokenizer.convert_tokens_to_string(tokens)
        final_words.append(text)
    return final_words


def attack(feature: Feature, target_model: BertForSequenceClassification,
           mlm_model: BertForMaskedLM, tokenizer: BertTokenizer, k: int,
           batch_size: int, max_length: int = 512,
           cos_mat: Optional[np.ndarray] = None, w2i: Optional[dict] = None,
           use_bpe: int = 1,
           threshold_pred_score: float = 0.3
           ):
    """Performs all substitution attacks on the BERT model."""
    # MLM-process
    if w2i is None:
        w2i = {}

    # Tokenize sentence for prediction
    words, sub_words, keys = tokenize(feature.seq, tokenizer)

    # Perform prediction on original sample.
    current_prob, orig_label, orig_predictions = predict_original(
        max_length, feature.seq, target_model, tokenizer
    )

    # If model fails on the original sentence, return.
    if orig_label != feature.label:
        feature.success = ORIG_MISCLASSIFIED
        return feature

    # Add special tokens and convert to IDs.
    sub_words = ['[CLS]'] + sub_words[:max_length - 2] + ['[SEP]']
    input_ids_ = torch.tensor([tokenizer.convert_tokens_to_ids(sub_words)])

    # Get top-k important words
    word_predictions = mlm_model(input_ids_.to('cuda'))[0].squeeze()
    word_pred_scores_all, word_predictions = torch.topk(word_predictions, k, -1)

    # Generate importance scores
    word_predictions = word_predictions[1:len(sub_words) + 1, :]
    word_pred_scores_all = word_pred_scores_all[1:len(sub_words) + 1, :]
    important_scores = generate_importance_scores(words, target_model,
                                                  current_prob, orig_label,
                                                  orig_predictions,
                                                  tokenizer, batch_size,
                                                  max_length)
    feature.query += int(len(words))
    list_of_index = sorted(enumerate(important_scores), key=lambda x: x[1],
                           reverse=True)  # Returns (word, importance) pairs
    final_words = copy.deepcopy(words)

    # For each word
    for top_index in list_of_index:
        # Load word and sub-word indices
        word_idx, word_score = top_index
        sub_word_i, sub_word_j = keys[word_idx]

        # Do not exceed 40% of the sentence.
        if feature.change > int(0.4 * (len(words))):
            feature.success = 1  # exceed
            return feature

        # Discard filter words.
        target_word = words[word_idx]
        if target_word in FILTER_WORDS:
            continue

        # Ignore the final period
        if sub_word_i > max_length - 2:
            continue

        # Load sub-words and prediction scores for the target words.
        substitutes = word_predictions[sub_word_i:sub_word_j]
        word_pred_scores = word_pred_scores_all[sub_word_i:sub_word_j]

        # Generate substitutes
        substitutes = get_substitutes(substitutes, tokenizer, mlm_model,
                                      use_bpe, word_pred_scores,
                                      threshold_pred_score)

        most_gap = 0.0
        candidate = None

        for substitute in substitutes:
            # Remove the word itself, sub-words, filter words and words that are
            # too different.
            if substitute == target_word or '##' in substitute or substitute in\
                    FILTER_WORDS or (substitute in w2i and target_word in w2i
                                     and cos_mat[w2i[substitute]]
                                     [w2i[target_word]] <
                                     0.4):
                continue

            # Generate text for prediction
            temp_replace = final_words
            temp_replace[word_idx] = substitute
            temp_text = tokenizer.convert_tokens_to_string(temp_replace)

            # Perform prediction
            temp_label, temp_prob = predict_target(target_model, tokenizer,
                                                   temp_text, max_length)

            # Increase the number of queries
            feature.query += 1

            if temp_label != orig_label:
                feature.change += 1
                final_words[word_idx] = substitute
                feature.changes.append(
                    [sub_word_i, substitute, target_word])
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
            feature.changes.append([sub_word_i, candidate, target_word])
            current_prob = current_prob - most_gap
            final_words[word_idx] = candidate

    feature.final_adverse = (tokenizer.convert_tokens_to_string(final_words))
    feature.success = 2
    return feature


def attack_infinite(feature: BigFeature,
                    target_model: BertForSequenceClassification,
                    mlm_model: BertForMaskedLM, tokenizer: BertTokenizer,
                    k: int,
                    batch_size: int, max_length: int = 512,
                    cos_mat: Optional[np.ndarray] = None,
                    w2i: Optional[dict] = None,
                    use_bpe: int = 1,
                    threshold_pred_score: float = 0.3
                    ) -> BigFeature:
    """Performs all substitution attacks on the BERT model."""
    # MLM-process
    if w2i is None:
        w2i = {}

    # Tokenize sentence for prediction
    words, sub_words, keys = tokenize(feature.seq, tokenizer)

    # Perform prediction on original sample.
    current_prob, orig_label, orig_predictions = predict_original(
        max_length, feature.seq, target_model, tokenizer
    )
    feature.orig_probs = orig_predictions.tolist()

    # Add special tokens and convert to IDs.
    sub_words = ['[CLS]'] + sub_words[:max_length - 2] + ['[SEP]']
    input_ids_ = torch.tensor([tokenizer.convert_tokens_to_ids(sub_words)])

    # Get top-k important words
    word_predictions = mlm_model(input_ids_.to('cuda'))[0].squeeze()
    word_pred_scores_all, word_predictions = torch.topk(word_predictions, k, -1)

    # Generate importance scores
    word_predictions = word_predictions[1:len(sub_words) + 1, :]
    word_pred_scores_all = word_pred_scores_all[1:len(sub_words) + 1, :]
    important_scores = generate_importance_scores(words, target_model,
                                                  current_prob, orig_label,
                                                  orig_predictions,
                                                  tokenizer, batch_size,
                                                  max_length)
    feature.query += int(len(words))
    list_of_index = sorted(enumerate(important_scores), key=lambda x: x[1],
                           reverse=True)  # Returns (word, importance) pairs
    final_words = copy.deepcopy(words)

    # For each word
    for top_index in list_of_index:
        # Load word and sub-word indices
        word_idx, word_score = top_index
        word_i, word_j = keys[word_idx]

        # Do not exceed 40% of the sentence.
        if feature.change > int(0.4 * (len(words))):
            feature.success = 1  # exceed
            return feature

        # Discard filter words.
        tgt_word = words[word_idx]
        if tgt_word in FILTER_WORDS:
            continue

        # Ignore the final period
        if word_i > max_length - 2:
            continue

        # Load sub-words and prediction scores for the target words.
        substitutes = word_predictions[word_i:word_j]
        word_pred_scores = word_pred_scores_all[word_i:word_j]

        # Generate substitutes
        substitutes = get_substitutes(substitutes, tokenizer, mlm_model,
                                      use_bpe, word_pred_scores,
                                      threshold_pred_score)

        most_gap = 0.0
        candidate = None

        for substitute in substitutes:
            # Remove the word itself, sub-words, filter words and words that are
            # too different.
            if \
                    substitute == tgt_word or \
                            '##' in substitute or \
                            substitute in FILTER_WORDS or \
                            (substitute in w2i and tgt_word in w2i and
                             cos_mat[w2i[substitute]][w2i[tgt_word]] < 0.4):
                continue

            # Generate text for prediction
            temp_replace = final_words
            temp_replace[top_index[0]] = substitute
            temp_text = tokenizer.convert_tokens_to_string(temp_replace)

            # Perform prediction
            temp_label, temp_prob = predict_target(target_model, tokenizer,
                                                   temp_text, max_length)

            # Update feature
            feature.adv_texts.append((tgt_word, substitute, temp_text))
            feature.adv_probs.append(temp_prob.tolist())

            # Increase the number of queries
            feature.query += 1

            if temp_label != orig_label:
                feature.change += 1
                final_words[top_index[0]] = substitute
                feature.changes.append(
                    [keys[top_index[0]][0], substitute, tgt_word])
                feature.final_adverse = temp_text
                feature.success = ATTACK_SUCCESSFUL
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


def predict_target(target_model, tokenizer, text, max_length):
    inputs = tokenizer.encode_plus(text, None,
                                   add_special_tokens=True,
                                   max_length=max_length,
                                   truncation='longest_first')
    input_ids = torch.tensor(inputs["input_ids"]).unsqueeze(0).to(
        'cuda')
    temp_prob = target_model(input_ids)[0].squeeze()
    temp_prob = torch.softmax(temp_prob, -1)
    temp_label = torch.argmax(temp_prob)
    return temp_label, temp_prob


def predict_original(max_length, text, tgt_model, tokenizer) -> (torch.Tensor,
                                                                 torch.Tensor,
                                                                 torch.Tensor):
    # Prepare feature for prediction
    inputs = tokenizer.encode_plus(
        text, None, add_special_tokens=True, max_length=max_length,
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
    return current_prob, orig_label, orig_predictions
