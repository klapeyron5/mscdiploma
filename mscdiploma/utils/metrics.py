import numpy as np
from nirvana.utils.tensorflow.imports import TfSetup
tf = TfSetup.import_tensorflow(3)
from nirvana.utils.tensorflow.utils import map_arrays


class Stat(dict):
    """
    author Yuri Litvinov https://ozforensics.slack.com/archives/CRH44KY5Q/p1576742908004200
    """
    def __init__(self, scores, labels, thr=0.5):
        super().__init__()

        assert len(scores) == len(labels)
        def type_check(arr):
            if type(arr) == list: arr = np.array(arr)
            else: assert type(arr) == np.ndarray
            return arr
        scores = type_check(scores)
        labels = type_check(labels)

        self.thr = thr
        scores[np.isnan(scores)] = -1
        idx = (scores >= 0) & ((labels == 0) | (labels == 1))
        self.exclusions = np.sum(1 - idx)
        labels = labels[idx]
        scores = scores[idx]
        self.total_n = np.sum(labels == 0)
        self.total_p = np.sum(labels == 1)
        self.fn = np.sum((labels == 1) & (scores < thr))
        self.tn = np.sum((labels == 0) & (scores < thr))
        self.fp = np.sum((labels == 0) & (scores >= thr))
        self.tp = np.sum((labels == 1) & (scores >= thr))
        self.mean_n = np.mean(scores[(labels == 0)])
        self.mean_p = np.mean(scores[(labels == 1)])
        self.far = self.fn / self.total_p if self.total_p > 0 else np.nan
        self.frr = self.fp / self.total_n if self.total_n > 0 else np.nan
        self.hter = (self.far + self.frr) / 2
        self.best_hter = 2
        self.best_hter_t = np.nan
        if len(labels) > 0:
            a = np.array(sorted(zip(scores, labels), reverse=True))[:, ::-1]
        else:
            a = np.zeros((0, 2))
        t = [0, 0]
        min_diff = 2
        old_far = 1
        old_frr = 0
        self.FAR = [old_far]
        self.FRR = [old_frr]
        self.ths = [1]
        self.eer_t = 2
        self.eer = -1
        self.auc = 0
        for i in range(a.shape[0]):
            t[int(a[i, 0])] += 1
            if (i < a.shape[0] - 1) and (a[i, 1] == a[i + 1, 1]):
                continue
            far_at_t = 1 - t[1] / self.total_p if self.total_p > 0 else np.nan
            frr_at_t = t[0] / self.total_n if self.total_n > 0 else np.nan
            diff = abs(far_at_t - frr_at_t)
            self.auc += (far_at_t + old_far) * (frr_at_t - old_frr) / 2
            old_far = far_at_t
            old_frr = frr_at_t
            self.FAR.append(far_at_t)
            self.FRR.append(frr_at_t)
            self.ths.append(a[i, 1])
            hter_at_t = (far_at_t + frr_at_t) / 2
            if hter_at_t < self.best_hter:
                self.best_hter = hter_at_t
                self.best_hter_t = a[i, 1]
            if diff > min_diff:
                continue
            min_diff = diff
            self.eer_t = a[i, 1]
            self.eer = (far_at_t + frr_at_t) / 2
        self.d_far = np.nan if self.fn < 5 else 1.96 * ((self.far * (1 - self.far)) / self.total_p) ** 0.5
        self.d_frr = np.nan if self.fp < 5 else 1.96 * ((self.frr * (1 - self.frr)) / self.total_n) ** 0.5
        self.d_hter = (self.d_far ** 2 + self.d_frr ** 2) ** 0.5
        for m in ("eer", "eer_t", "auc", "hter"):
            self[m] = getattr(self, m)


def softmax_loss(labels, logits):
    losses = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    return losses


def batch_softmax_loss(labels, logits):
    losses = softmax_loss(labels, logits)
    batch_loss = tf.reduce_mean(losses)
    return batch_loss


def softmax_weighted_loss(labels, logits, weights):
    out = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    w = labels*weights
    w_to_use = tf.argmax(w, axis=1)
    def f(weight, weight_to_use):
        return weight[weight_to_use]
    w = map_arrays(f, [w, w_to_use])

    out_clean = tf.reduce_mean(out)
    out_weighted = tf.reduce_mean(out*w)
    return out_weighted, out_clean


def acc(labels, predicts):
    out = tf.keras.metrics.binary_accuracy(labels, predicts, threshold=0.5)
    out = tf.reduce_mean(out)
    return out
