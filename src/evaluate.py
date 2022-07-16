import tensorflow.compat.v1 as tf
import tensorflow_hub as hub


class USE(object):
    """Class for generating sentence similarities.

    This class uses Universal Sentence Encoder (USE) and loads it from the cache path.
    The code uses Tensorflow v1.
    """
    def __init__(self, cache_path):
        """Loads and initializes USE and initializes unassigned variables."""
        super(USE, self).__init__()

        self.embed = hub.Module(cache_path)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session()
        self.build_graph()
        self.sess.run([tf.global_variables_initializer(),
                       tf.tables_initializer()])
        self.sts_input1 = None
        self.sts_input2 = None
        self.cosine_similarities = None
        self.sim_scores = None

    def build_graph(self):
        """Loads the tensorflow execution graph for similarity calculation."""
        self.sts_input1 = tf.placeholder(tf.string, shape=None)
        self.sts_input2 = tf.placeholder(tf.string, shape=None)

        sts_encode1 = tf.nn.l2_normalize(self.embed(self.sts_input1), axis=1)
        sts_encode2 = tf.nn.l2_normalize(self.embed(self.sts_input2), axis=1)
        self.cosine_similarities = tf.reduce_sum(
            tf.multiply(sts_encode1, sts_encode2), axis=1)
        clip_cosine_similarities = tf.clip_by_value(self.cosine_similarities,
                                                    -1.0, 1.0)
        self.sim_scores = 1.0 - tf.acos(clip_cosine_similarities)

    def semantic_sim(self, sentences1, sentences2):
        """Calculates similarity between two sentences."""
        sentences1 = [s.lower() for s in sentences1]
        sentences2 = [s.lower() for s in sentences2]
        scores = self.sess.run(
            [self.sim_scores],
            feed_dict={
                self.sts_input1: sentences1,
                self.sts_input2: sentences2,
            })
        return scores[0]


def evaluate(features):
    """Evaluates the model performance.

    It calculates three key metrics:
        1. Change rate: The number of words before success.
        2. Accuracy: The accuracy of the model after attack.
        3. Query number: The average number of queries to the first success.
    """
    do_use = 0
    use = None
    sim_threshold = 0

    # evaluate with USE
    if do_use == 1:
        cache_path = 'models/use'
        use = USE(cache_path)

    acc = 0
    origin_success = 0
    total = 0
    total_q = 0
    total_change = 0
    total_word = 0
    for feat in features:
        if feat.success > 2:

            if do_use == 1:
                sim = float(use.semantic_sim([feat.seq], [feat.final_adverse]))
                if sim < sim_threshold:
                    continue

            acc += 1
            total_q += feat.query
            total_change += feat.change
            total_word += len(feat.seq.split(' '))

            if feat.success == 3:
                origin_success += 1

        total += 1

    suc = float(acc / total)

    query = float(total_q / acc)
    change_rate = float(total_change / total_word)

    origin_acc = 1 - origin_success / total
    after_atk = 1 - suc

    print('acc/aft-atk-acc {:.6f}/ {:.6f}, query-num {:.4f}, change-rate {:.4f}'
          ''.format(origin_acc, after_atk, query, change_rate))

    return after_atk, query, change_rate
