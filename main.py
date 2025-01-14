# Adapted from https://github.com/LinyangLee/BERT-Attack
#
# Original information as follows:
# -*- coding: utf-8 -*-
# @Time    : 2020/6/10
# @Author  : Linyang Li
# @Email   : linyangli19@fudan.edu.cn
# @File    : attack.py
import pickle
import sys
import warnings
import os
import torch
import argparse
from src.attack import attack, attack_infinite
from src.evaluate import evaluate
from src.utils import load_similarity_embed, load_dataset, Feature, \
    dump_features, load_models, BigFeature

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)


def main(arguments: dict, sim_directory: str = None):
    print('Starting process...')
    # Load required models, tokenizers, and dataset
    model_mlm, model_target, tokenizer_target = load_models(arguments)
    samples = load_dataset(arguments["data_path"])

    # Load the similarity matrix
    if arguments["use_sim_mat"] == 1:
        if sim_directory is None:
            sim_directory = "data_defense"
        print('\tLoading similarity embeddings...')
        cos_mat, w2i, i2w = load_similarity_embed(
            sim_directory + '/counter-fitted-vectors.txt',
            sim_directory + '/cos_sim_counter_fitting.npy',
        )
    else:
        cos_mat, w2i, i2w = None, {}, {}

    print("\tPerforming attacks...")
    features_output = []
    with torch.no_grad():
        print()
        n = arguments["end"] - arguments["start"]
        for index, feature in enumerate(
                samples[arguments["start"]:arguments["end"]]
        ):
            # Convert sample to object with embedded metrics
            feature = Feature(*feature)

            # Perform attacks on the sample
            print('\r\t\t[{:d} / {:d}] '.format(index, n), end='')
            feature = attack(
                feature, model_target, model_mlm, tokenizer_target,
                arguments["k"], batch_size=32, max_length=512, cos_mat=cos_mat,
                w2i=w2i, use_bpe=arguments["use_bpe"],
                threshold_pred_score=arguments["threshold_pred_score"],
                word_sim=arguments["word_sim"]
            )

            if feature.success > 2:
                print('Successful', end='')
            else:
                print('Failed', end='')
            features_output.append(feature)

    # Evaluate and save
    print("\r\tEvaluating performance")
    after_atk, query, change_rate = evaluate(features_output)

    print("\r\tSaving files")
    dump_features(features_output, arguments["output_dir"])

    print("Completed")

    return after_atk, query, change_rate


def dump_features_infinite(features, out_path):
    with open(os.path.join(out_path, 'infinite.pickle'), 'wb') as file:
        pickle.dump(features, file)


def main_infinite(arguments: dict, sim_directory: str = None):
    print('Starting process...')
    # Load required models, tokenizers, and dataset
    model_mlm, model_target, tokenizer_target = load_models(arguments)
    samples = load_dataset(arguments["data_path"])

    # Load the similarity matrix
    if arguments["use_sim_mat"] == 1:
        if sim_directory is None:
            sim_directory = "data_defense"
        print('\tLoading similarity embeddings...')
        cos_mat, w2i, i2w = load_similarity_embed(
            sim_directory + '/counter-fitted-vectors.txt',
            sim_directory + '/cos_sim_counter_fitting.npy',
        )
    else:
        cos_mat, w2i, i2w = None, {}, {}

    print("\tPerforming attacks...")
    features_output = []
    with torch.no_grad():
        print()
        n = arguments["end"] - arguments["start"]
        for index, feature in enumerate(
                samples[arguments["start"]:arguments["end"]]
        ):
            # Convert sample to object with embedded metrics
            feature = BigFeature(*feature)

            # Perform attacks on the sample
            print('\r\t\t[{:d} / {:d}] '.format(index, n), end='')
            feature = attack_infinite(
                feature, model_target, model_mlm, tokenizer_target,
                arguments["k"], batch_size=32, max_length=512, cos_mat=cos_mat,
                w2i=w2i, use_bpe=arguments["use_bpe"],
                threshold_pred_score=arguments["threshold_pred_score"],
                word_sim=arguments["word_sim"]
            )

            if feature.success > 2:
                print('Successful', end='')
            else:
                print('Failed', end='')
            features_output.append(feature)

    print("\r\tSaving files")
    dump_features_infinite(features_output, arguments["output_dir"])

    print("Completed")


def parse_args() -> dict:
    """Load and parse arguments.

    Converts each

    The parser accepts the following arguments:
        data_path: Path to the TSV dataset file.
        mlm_path: Path to the directory containing the MLM model.
        tgt_path: Path to the directory containing the classifier.
        output_dir: Path to experiment results. Must be a directory.
        use_sim_mat: Flag to set if cosine similarity is used to filter
            antonyms.
        start: Starting step. Usable for multi-threaded processing.
        start: Ending step. Usable for multi-threaded processing.
        num_label: Number of labels.
        use_bpe: Decides whether to use byte-pair encoding or not.
        k: Number of words to be tested for replacement.
        threshold_pred_score: Positive prediction threshold.
    """
    # Parse all arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str,
                        help="Path to the TSV dataset file.")
    parser.add_argument("--mlm_path", type=str,
                        help="Path to the directory containing the MLM model.")
    parser.add_argument("--tgt_path", type=str,
                        help="Path to the directory containing the classifier.")
    parser.add_argument("--output_dir", type=str,
                        help="Path to experiment results. Must be a directory.")
    parser.add_argument("--use_sim_mat", type=int,
                        help='Flag to set if cosine similarity is used to '
                             'filter antonyms.')
    parser.add_argument("--start", type=int,
                        help="Starting step. Usable for multi-threaded "
                             "processing.")
    parser.add_argument("--end", type=int,
                        help="Ending step. Usable for multi-threaded "
                             "processing.")
    parser.add_argument("--num_label", type=int, help="Number of labels")
    parser.add_argument("--use_bpe", type=int, help="")
    parser.add_argument("--k", type=int,
                        help="Number of words to be tested for replacement.")
    parser.add_argument("--threshold_pred_score", type=float,
                        help="Positive prediction threshold.")
    parser.add_argument("--word_sim", type=float,
                        help="Word similarity.")
    arguments = parser.parse_args()

    # Return namespace
    return vars(arguments)


if __name__ == '__main__':
    # Parse arguments
    if len(sys.argv) < 11:
        args = {
            "data_path": "data_defense/imdb_1k.tsv",
            "mlm_path": "./bert-base-uncased",
            "tgt_path": "./models/imdbclassifier",
            "use_sim_mat": 0,
            "output_dir": "data_defense",
            "num_label": 2,
            "use_bpe": 1,
            "k": 48,
            "start": 0,
            "end": 1,
            "threshold_pred_score": 0,
            "word_sim": 0.4,
        }
    else:
        args = parse_args()
    main_infinite(args)
