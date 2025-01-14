import copy
from typing import List, Tuple, Optional
import numpy as np
import torch
from torch import nn as nn
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
from transformers import BertForSequenceClassification, BertTokenizer, \
    BertForMaskedLM
from src.utils import Feature, ORIG_MISCLASSIFIED, BigFeature, \
    ATTACK_SUCCESSFUL, EXCEED_40, ATTACK_EXCEED_40, ATTACK_UNSUCCESSFUL

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

    This is performed by masking each word in the sentence and checking the
    difference in output probabilities.

    For example, consider a sentiment analysis task on a simple sentence before
    and after masking.

    P('Happy' | "I am very happy.") = 0.9
    P('Happy' | "I am very [MASK]") = 0.1
    Importance("happy") = 0.9 - 0.1 = 0.8

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
        [SEP] token.
    """
    # Create a list of masked sentences. Every word except the final [SEP] token
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

    # Calculate one importance score for each word. This is calculated using
    # the sum of two parts:
    #   1. Difference between the correct label probabilities
    #   2. The difference between the incorrect label probabilities, if
    #      misclassified.
    import_scores = orig_prob - probs[:, orig_label] + (probs_argmax !=
                                                        orig_label).float() * (
            probs.max(dim=-1)[0] - torch.index_select(orig_probs, 0,
                                                      probs_argmax))
    import_scores = import_scores.data.cpu().numpy()  # num_masked

    print("Original prob" + str(orig_prob))

    return import_scores


def get_substitutes(substitutes, tokenizer, mlm_model, use_bpe,
                    substitutes_score=None, threshold=3.0):
    """Creates a list of substitute words as a set of strings.

    This method simply filters suggested substitutes by their importance scores
    and converts them from indices to strings. If they are sub-words, however,
    a more complex process is followed as described in get_bpe_substitutes.

    An example of a sub word is:
        Let's = Let + ' + s

    Args:
        substitutes: A list of indices of suggested substitutes for the word.
        tokenizer: A tokenizer for converting words back to tokens.
        mlm_model: A BERT instance to be used if a word consists of sub-words
        use_bpe: Checks if sub-words are to be replaced.
        substitutes_score: A list of scores corresponding to substitutes.
        threshold: A threshold for importance scores for words to be considered.

    Returns:

    """
    words = []
    sub_len, k = substitutes.size()  # sub-len, k

    # If the word is empty, return.
    if sub_len == 0:
        return words

    # If it only contains a single sub-word, i.e., it's a complete word,
    # check all the substitutes for the word and convert them to strings.
    elif sub_len == 1:
        for (i, j) in zip(substitutes[0], substitutes_score[0]):
            # If it's a valid word and is unimportant, discard it and return.
            if threshold != 0 and j < threshold:
                break
            words.append(tokenizer._convert_id_to_token(int(i)))

    # Otherwise, we know that there are multiple sub-words in the sentence.
    # If sub-words are allowed to be used, check their substitutes. Otherwise,
    # return.
    else:
        if use_bpe == 1:
            words = get_bpe_substitutes(substitutes, tokenizer, mlm_model)
        else:
            return words

    return words


def get_bpe_substitutes(substitutes, tokenizer, mlm_model):
    """Generates substitutes for sub-words.

    The input is a set of substitutes, k substitutes for each sub-word.
    Consider an example of the sentence "Let's go." Assuming that we select the
    word "Let's", we get 48 words for "Let", "'", and "s" each, creating a
    vector of size (3 x 48).

    We create some N combinations of these substitutes, obtaining N triplets.
    The resulting vector has a size (N x 3).

    Now, we calculate the perplexity of each triplet. Perplexity is roughly
    the opposite of the likelihood of encountering a combination of words.
    The lower the likelihood of the triplet, the more likely it is to be
    encountered naturally.

    Finally, these combinations are ranked, stringified and returned.

    Args:
        substitutes: The set of substitutes for each sub-word.
        tokenizer: BERT tokenizer for index to string conversion.
        mlm_model: Language model to calculate perplexity.

    Returns:
        List of substitute words for the input set of sub-words.
    """
    # k suggested substitutes for L sub-words.
    substitutes = substitutes[0:12, 0:4]  # maximum BPE candidates

    # Generate all the possible combinations of suggested sub-words, similar to
    # a cartesian product.
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

    # Initialize loss function to calculate perplexity.
    c_loss = nn.CrossEntropyLoss(reduction='none')

    # Convert to a tensor and reduce the number of combinations. The resulting
    # tensor contains N sets consisting of L words each. So, every element is
    # a set of sub-words replacing the original word.
    all_substitutes = torch.tensor(all_substitutes)  # [ N, L ]
    all_substitutes = all_substitutes[:24].to('cuda')

    N, L = all_substitutes.size()

    # Get probable words for all combinations
    word_predictions = mlm_model(all_substitutes)[0]  # N L vocab-size

    # Calculate the perplexity for all words combinations using selective loss
    # like cross entropy. The operation calculates perplexity as a loss for
    # every triplet.
    ppl = c_loss(word_predictions.view(N * L, -1),
                 all_substitutes.view(-1))  # [ N*L ]
    ppl = torch.exp(torch.mean(ppl.view(N, L), dim=-1))  # N

    # Sort the word substitutes sets by their perplexity
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
           threshold_pred_score: float = 0.3, word_sim: float = 0.4
           ):
    """Performs all substitution attacks on the BERT model.

    This function performs attacks on a single sentence. The process is as
    follows:
        1. Pre-process the sentence.
        2. Consider every word in the sentence and judge how important it is to
        the classifier, which is guessed by how much the output is changed by
        the absence of this word.
        3. If the classifier misclassifies the sentence, stop.
        3. Sort these words based on their importance and pick the top k words
        to replace.
        4. For each word, do the following:
            i.   Generate a substitute word using the language model BERT. This
            is performed by hiding the word and asking BERT to predict it.
            ii.   If more than 40% words in the sentence have been tried, stop.
            iii. Otherwise, generate substitutes.
            iv.  For each substitute, check if the modified sentence
            successfully tricks the classifier. If so, end. Otherwise, keep
            trying.
        5. If no attacks are successful, return the failure.


    Args:
        feature: An initialized Feature object. Only the sequence is assigned
        a valid value.
        target_model: The victim classifier.
        mlm_model: The MLM BERT instance.
        k: The number of words to be considered while attacking.
        batch_size: The batch size while generating importance scores.
        max_length: The length to which a sentence should be padded for a
        uniform size.
        cos_mat: The cosine similarity matrix to compare words.
        w2i: A map converting word strings to indices for cos_mat.
        use_bpe: Decides whether substitutes are to be generated for tokens
        smaller than a word.
        threshold_pred_score: Decides whether a sub-word is to be discarded when
        generating substitutions. It is compared to the sub-word's importance
        score.
        word_sim: Controls how close the words must be. If the cosine similarity
        of a substitute is greater than word_sim, it is discarded.

    Returns:
        A utility class Feature containing attack and sentence statistics.
    """
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
            feature.success = ATTACK_EXCEED_40  # exceed
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
                                     word_sim):
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
                feature.success = ATTACK_SUCCESSFUL
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
    feature.success = ATTACK_UNSUCCESSFUL
    return feature


def attack_infinite(feature: BigFeature,
                    target_model: BertForSequenceClassification,
                    mlm_model: BertForMaskedLM, tokenizer: BertTokenizer,
                    k: int,
                    batch_size: int, max_length: int = 512,
                    cos_mat: Optional[np.ndarray] = None,
                    w2i: Optional[dict] = None,
                    use_bpe: int = 1,
                    threshold_pred_score: float = 0.3,
                    word_sim: float = 0.4
                    ) -> BigFeature:
    """Performs all substitution attacks on the BERT model.

    Most of the implementation is similar to `attack`. There are only two
    modifications made:
        1. The attacks aren't stopped after the first success. All attacks are
        performed until 40% of the words in the sentence have been tried.
        2. Even if the original classifier misclassifies the sample, the attacks
        are continued.
        3. For each individual substitution attack, the following information is
        stored:
            a. Original word
            b. Substituted word
            c. Adversarial text
            d. Classifier probabilities on adversarial text.

    Args:
        feature: An initialized BigFeature object. Only the sequence is assigned
        a valid value.
        target_model: The victim classifier.
        mlm_model: The MLM BERT instance.
        k: The number of words to be considered while attacking.
        batch_size: The batch size while generating importance scores.
        max_length: The length to which a sentence should be padded for a
        uniform size.
        cos_mat: The cosine similarity matrix to compare words.
        w2i: A map converting word strings to indices for cos_mat.
        use_bpe: Decides whether substitutes are to be generated for tokens
        smaller than a word.
        threshold_pred_score: Decides whether a sub-word is to be discarded when
        generating substitutions. It is compared to the sub-word's importance
        score.
        word_sim: Controls how close the words must be. If the cosine similarity
        of a substitute is greater than word_sim, it is discarded.

    Returns:
        A utility class BigFeature containing attack and sentence statistics.
    """
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
            feature.success = ATTACK_EXCEED_40  # exceed
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
                             cos_mat[w2i[substitute]][w2i[tgt_word]] < word_sim):
                continue

            # Generate text for prediction
            temp_replace = final_words
            temp_replace[word_idx] = substitute
            temp_text = tokenizer.convert_tokens_to_string(temp_replace)

            # Generate masked sentence
            masked_sentence = [x for x in final_words]
            masked_sentence[word_idx] = tokenizer.mask_token
            masked_sentence = tokenizer.convert_tokens_to_string(masked_sentence)

            # Perform prediction
            temp_label, temp_prob = predict_target(target_model, tokenizer,
                                                   temp_text, max_length)

            # Update feature
            feature.adv_texts.append((tgt_word, substitute, masked_sentence,
                                      temp_text))
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
    feature.success = ATTACK_UNSUCCESSFUL
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
