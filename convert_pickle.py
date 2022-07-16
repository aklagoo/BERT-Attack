import pickle
import numpy as np
import pandas as pd

if __name__ == '__main__':
    with \
            open('data_defense/infinite.pickle', 'rb') as pickle_file, \
            open('attacked_data/orig.txt', 'w', encoding='utf-8') as orig_file, \
            open('attacked_data/orig_preds.txt', 'w', encoding='utf-8') as orig_pred_file, \
            open('attacked_data/label.txt', 'w', encoding='utf-8') as label_file, \
            open('attacked_data/adv.txt', 'w', encoding='utf-8') as adv_file, \
            open('attacked_data/adv_single.txt', 'w', encoding='utf-8') as adv_single_file, \
            open('attacked_data/adv_preds.txt', 'w', encoding='utf-8') as adv_single_preds_file, \
            open('attacked_data/index.txt', 'w') as index_file,\
            open('attacked_data/statistics.csv', 'w', encoding='utf-8') as stats_file:
        obj = pickle.load(pickle_file)

        stats = {
            'success_rate': [],
            'first_correct': [],
            'max_success_idx': [],
            'max_success_prob': [],
            'max_success_text': [],
        }
        for feature in obj:
            # Calculate statistics
            adv_preds = np.argmax(feature.adv_probs, axis=1)
            comparisons = adv_preds != feature.label
            first_correct = int(np.argmax(comparisons))
            accuracy = comparisons.sum() / comparisons.shape[0]
            probs = np.array(feature.adv_probs)

            max_success_idx = np.argmin(probs[:, feature.label])
            max_success_prob = probs[max_success_idx, feature.label]
            max_success_probs = probs[max_success_idx]
            max_success_text = feature.adv_texts[max_success_idx][2]

            stats['success_rate'].append(accuracy)
            stats['first_correct'].append(first_correct)
            stats['max_success_idx'].append(max_success_idx)
            stats['max_success_prob'].append(max_success_prob)
            stats['max_success_text'].append(max_success_text)

            # Write text to file
            orig_file.write(feature.seq + "\n")
            label_file.write(str(feature.label) + "\n")
            index_file.write(str(len(feature.adv_texts)))
            for _, _, adv_text in feature.adv_texts:
                adv_file.write(adv_text + "\n")
            adv_single_file.write(max_success_text + "\n")
            orig_pred_file.write("{0},{1}\n".format(*feature.orig_probs))
            adv_single_preds_file.write("{0},{1}\n".format(max_success_probs[0],
                                                           max_success_probs[1])
                                        )

        pd.DataFrame(stats).to_csv(stats_file)
