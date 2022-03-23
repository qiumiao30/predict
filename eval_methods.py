import numpy as np
import more_itertools as mit
from spot import SPOT, dSPOT
from args import get_parser
parser = get_parser()
args = parser.parse_args()


def adjust_predicts(score, label, threshold, pred=None, calc_latency=False):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.
    Args:
            score (np.ndarray): The anomaly score
            label (np.ndarray): The ground-truth label
            threshold (float): The threshold of anomaly score.
                    A point is labeled as "anomaly" if its score is lower than the threshold.
            pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
            calc_latency (bool):
    Returns:
            np.ndarray: predict labels

    Method from OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly)
    """
    if label is None:
        predict = score > threshold
        return predict, None

    if pred is None:
        if len(score) != len(label):
            raise ValueError("score and label must have the same length")
        predict = score > threshold # true or flase, length=(len(test)-windows)
    else:
        predict = pred

    actual = label > 0.1 # 与label相同
    anomaly_state = False
    anomaly_count = 0
    latency = 0

    for i in range(len(predict)):
        if any(actual[max(i, 0) : i + 1]) and predict[i] and not anomaly_state: # 真实标签为True，且预测标签为True
            anomaly_state = True
            anomaly_count += 1
            for j in range(i, 0, -1):
                if not actual[j]: # 当真实标签为False时，执行break,真实标签为True，则不成立.
                    break
                else:
                    if not predict[j]: # 预测标签为False，然后将预测值设为True，即真实标签为True，预测标签为False，然后将预测标签标记为True
                        predict[j] = True
                        latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict

def pak(scores, targets, thres, k):
    """
    :param scores: anomaly scores
    :param targets: target labels
    :param thres: anomaly threshold
    :param k: PA%K ratio, 0 equals to conventional point adjust and 100 equals to original predictions
    :return: point_adjusted predictions
    """
    scores = np.array(scores)
    thres = np.array(thres)

    predicts = scores > thres
    actuals = targets > 0.01

    one_start_idx = np.where(np.diff(actuals, prepend=0) == 1)[0]
    zero_start_idx = np.where(np.diff(actuals, prepend=0) == -1)[0]

    assert len(one_start_idx) == len(zero_start_idx) + 1 or len(one_start_idx) == len(zero_start_idx)

    if len(one_start_idx) == len(zero_start_idx) + 1:
        zero_start_idx = np.append(zero_start_idx, len(predicts))

    for i in range(len(one_start_idx)):
        if predicts[one_start_idx[i]:zero_start_idx[i]].sum() > k / 100 * (zero_start_idx[i] - one_start_idx[i]):
            predicts[one_start_idx[i]:zero_start_idx[i]] = 1

    return predicts

def calc_point2point(predict, actual):
    """
    calculate f1 score by predict and actual.
    Args:
            predict (np.ndarray): the predict label
            actual (np.ndarray): np.ndarray
    Method from OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly)
    """
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
    return f1, precision, recall, TP, TN, FP, FN


def pot_eval(init_score, score, label, k, q=1e-3, level=0.99, dynamic=False, adjust_score=args.adjust_score):
    """
    Run POT method on given score.
    :param init_score (np.ndarray): The data to get init threshold.
                    For `OmniAnomaly`, it should be the anomaly score of train set.
    :param: score (np.ndarray): The data to run POT method.
                    For `OmniAnomaly`, it should be the anomaly score of test set.
    :param label (np.ndarray): boolean list of true anomalies in score
    :param q (float): Detection level (risk)
    :param level (float): Probability associated with the initial threshold t
    :return dict: pot result dict
    Method from OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly)
    """

    print(f"Running POT with q={q}, level={level}..")
    s = dSPOT(q)  # SPOT object
    s.fit(init_score, score)
    s.initialize() # DSPOT
    # s.initialize(level=level, min_extrema=False)  # Calibration step，SPOT
    ret = s.run(with_alarm=False) # DSPOT
    # ret = s.run(dynamic=dynamic, with_alarm=False) # SPOT

    print("alarms length: ", len(ret["alarms"]))
    print("thresholds length: ", len(ret["thresholds"]))


    pot_th = np.mean(ret["thresholds"])
    # if adjust_score:
    #     pred, p_latency = adjust_predicts(score, label, pot_th, calc_latency=True)
    #     predict, k = pak(score, label, threshold, k), k
    if adjust_score:
        pred, k = pak(score, label, pot_th, k), k
    else:
        pred, k = score, 0
    if label is not None:
        p_t = calc_point2point(pred, label)
        return {
            "f1": p_t[0],
            "precision": p_t[1],
            "recall": p_t[2],
            "TP": p_t[3],
            "TN": p_t[4],
            "FP": p_t[5],
            "FN": p_t[6],
            "threshold": pot_th,
            "k(k=0-->PA,k=100-->original)": k,
        }
    else:
        return {
            "threshold": pot_th,
        }


def bf_search(score, label, k, start, end=None, step_num=1, display_freq=10, verbose=True):
    """
    Find the best-f1 score by searching best `threshold` in [`start`, `end`).
    Method from OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly)
    """

    print(f"Finding best f1-score by searching for threshold..")
    if step_num is None or end is None:
        end = start
        step_num = 1
    search_step, search_range, search_lower_bound = step_num, end - start, start
    if verbose:
        print("search range: ", search_lower_bound, search_lower_bound + search_range)
    threshold = search_lower_bound
    m = (-1.0, -1.0, -1.0)
    m_t = 0.0
    m_l = 0
    for i in range(search_step):
        threshold += search_range / float(search_step)
        target, k = calc_seq(score, label, threshold, k)
        # pak =
        if target[0] > m[0]:
            m_t = threshold
            m = target
            m_l = k
        if verbose and i % display_freq == 0:
            print("cur thr: ", threshold, target, m, m_t)

    return {
        "f1": m[0],
        "precision": m[1],
        "recall": m[2],
        "TP": m[3],
        "TN": m[4],
        "FP": m[5],
        "FN": m[6],
        "threshold": m_t,
        "k(k=0-->PA,k=100-->original)": m_l,
    }


# def calc_seq(score, label, threshold, adjust_score=args.adjust_score):
#     if adjust_score:
#         predict, latency = adjust_predicts(score, label, threshold, calc_latency=True)
#     else:
#         predict, latency = (score >= threshold).astype(int), 0
#     return calc_point2point(predict, label), latency

def calc_seq(score, label, threshold, k, adjust_score=args.adjust_score):
    if adjust_score:
        predict, k = pak(score, label, threshold, k), k
    else:
        predict, k = (score >= threshold).astype(int), 0
    return calc_point2point(predict, label), k


def epsilon_eval(train_scores, test_scores, test_labels, k, reg_level=1, adjust_score=args.adjust_score):
    best_epsilon = find_epsilon(train_scores, reg_level)
    # if adjust_score:
    #     pred, p_latency = adjust_predicts(test_scores, test_labels, best_epsilon, calc_latency=True)
    if adjust_score:
        pred, k = pak(test_scores, test_labels, best_epsilon, k), k
    else:
        pred, k = (test_scores >= best_epsilon).astype(int), k
    if test_labels is not None:
        p_t = calc_point2point(pred, test_labels)
        return {
            "f1": p_t[0],
            "precision": p_t[1],
            "recall": p_t[2],
            "TP": p_t[3],
            "TN": p_t[4],
            "FP": p_t[5],
            "FN": p_t[6],
            "threshold": best_epsilon,
            "k(k=0-->PA,k=100-->original)": k,
            "reg_level": reg_level,
        }
    else:
        return {"threshold": best_epsilon, "reg_level": reg_level}


def find_epsilon(errors, reg_level=1):
    """
    Threshold method proposed by Hundman et. al. (https://arxiv.org/abs/1802.04431)
    Code from TelemAnom (https://github.com/khundman/telemanom)
    """
    e_s = errors
    best_epsilon = None
    max_score = -10000000
    mean_e_s = np.mean(e_s)
    sd_e_s = np.std(e_s)

    for z in np.arange(2.5, 12, 0.5):
        epsilon = mean_e_s + sd_e_s * z
        pruned_e_s = e_s[e_s < epsilon]

        i_anom = np.argwhere(e_s >= epsilon).reshape(-1,)
        buffer = np.arange(1, 50)
        i_anom = np.sort(
            np.concatenate(
                (
                    i_anom,
                    np.array([i + buffer for i in i_anom]).flatten(),
                    np.array([i - buffer for i in i_anom]).flatten(),
                )
            )
        )
        i_anom = i_anom[(i_anom < len(e_s)) & (i_anom >= 0)]
        i_anom = np.sort(np.unique(i_anom))

        if len(i_anom) > 0:
            groups = [list(group) for group in mit.consecutive_groups(i_anom)]
            # E_seq = [(g[0], g[-1]) for g in groups if not g[0] == g[-1]]

            mean_perc_decrease = (mean_e_s - np.mean(pruned_e_s)) / mean_e_s
            sd_perc_decrease = (sd_e_s - np.std(pruned_e_s)) / sd_e_s
            if reg_level == 0:
                denom = 1
            elif reg_level == 1:
                denom = len(i_anom)
            elif reg_level == 2:
                denom = len(i_anom) ** 2

            score = (mean_perc_decrease + sd_perc_decrease) / denom

            if score >= max_score and len(i_anom) < (len(e_s) * 0.5):
                max_score = score
                best_epsilon = epsilon

    if best_epsilon is None:
        best_epsilon = np.max(e_s)
    return best_epsilon