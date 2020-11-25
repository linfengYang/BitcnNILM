from Logger import log
import numpy as np
import pandas as pd


# from sklearn.metrics import confusion_matrix

def get_TP(target, prediction, threshold):
    '''
    compute the  number of true positive

    Parameters:
    ----------------
    target: the groud truth , np.array
    prediction: the prediction, np.array
    threshold: float
    '''

    assert (target.shape == prediction.shape)

    target = 1 - np.clip(target, threshold, 0) / threshold
    prediction = 1 - np.clip(prediction, threshold, 0) / threshold

    tp_array = np.logical_and(target, prediction) * 1.0
    tp = np.sum(tp_array)

    return tp


def get_FP(target, prediction, threshold):
    '''
    compute the  number of false positive

    Parameters:
    ----------------
    target: the groud truth , np.array
    prediction: the prediction, np.array
    threshold: float
    '''

    assert (target.shape == prediction.shape)

    target = np.clip(target, threshold, 0) / threshold
    prediction = 1 - np.clip(prediction, threshold, 0) / threshold

    fp_array = np.logical_and(target, prediction) * 1.0
    fp = np.sum(fp_array)

    return fp


def get_FN(target, prediction, threshold):
    '''
    compute the  number of false negtive

    Parameters:
    ----------------
    target: the groud truth , np.array
    prediction: the prediction, np.array
    threshold: float
    '''

    assert (target.shape == prediction.shape)

    target = 1 - np.clip(target, threshold, 0) / threshold
    prediction = np.clip(prediction, threshold, 0) / threshold

    fn_array = np.logical_and(target, prediction) * 1.0
    fn = np.sum(fn_array)

    return fn


def get_TN(target, prediction, threshold):
    '''
    compute the  number of true negative

    Parameters:
    ----------------
    target: the groud truth , np.array
    prediction: the prediction, np.array
    threshold: float
    '''

    assert (target.shape == prediction.shape)

    target = np.clip(target, threshold, 0) / threshold
    prediction = np.clip(prediction, threshold, 0) / threshold

    tn_array = np.logical_and(target, prediction) * 1.0
    tn = np.sum(tn_array)

    return tn


def get_recall(target, prediction, threshold):
    '''
    compute the recall rate

    Parameters:
    ----------------
    target: the groud truth , np.array
    prediction: the prediction, np.array
    threshold: float
    '''

    tp = get_TP(target, prediction, threshold)
    fn = get_FN(target, prediction, threshold)
    log('tp={0}'.format(tp))
    log('fn={0}'.format(fn))
    if tp + fn <= 0.0:
        recall = tp / (tp + fn + 1e-9)
    else:
        recall = tp / (tp + fn)
    return recall


def get_precision(target, prediction, threshold):
    '''
    compute the  precision rate

    Parameters:
    ----------------
    target: the groud truth , np.array
    prediction: the prediction, np.array
    threshold: float
    '''

    tp = get_TP(target, prediction, threshold)
    fp = get_FP(target, prediction, threshold)
    log('tp={0}'.format(tp))
    log('fp={0}'.format(fp))
    if tp + fp <= 0.0:
        precision = tp / (tp + fp + 1e-9)
    else:
        precision = tp / (tp + fp)
    return precision


def get_F1(target, prediction, threshold):
    '''
    compute the  F1 score

    Parameters:
    ----------------
    target: the groud truth , np.array
    prediction: the prediction, np.array
    threshold: float
    '''

    recall = get_recall(target, prediction, threshold)
    log(recall)
    precision = get_precision(target, prediction, threshold)
    log(precision)
    if precision == 0.0 or recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return f1


def get_accuracy(target, prediction, threshold):
    '''
    compute the accuracy rate

    Parameters:
    ----------------
    target: the groud truth , np.array
    prediction: the prediction, np.array
    threshold: float
    '''

    tp = get_TP(target, prediction, threshold)
    tn = get_TN(target, prediction, threshold)

    accuracy = (tp + tn) / target.size

    return accuracy


def get_relative_error(target, prediction):
    '''
    compute the  relative_error

    Parameters:
    ----------------
    target: the groud truth , np.array
    prediction: the prediction, np.array
    '''

    assert (target.shape == prediction.shape)

    return np.mean(np.nan_to_num(np.abs(target - prediction) / np.maximum(target, prediction)))


def get_abs_error(target, prediction):
    '''
    compute the  absolute_error

    Parameters:
    ----------------
    target: the groud truth , np.array
    prediction: the prediction, np.array
    '''

    assert (target.shape == prediction.shape)

    data = np.abs(target - prediction)
    mean, std, min_v, max_v, quartile1, median, quartile2 = get_statistics(data)

    return mean, std, min_v, max_v, quartile1, median, quartile2, data


def get_nde(target, prediction):
    '''
    compute the  normalized disaggregation error

    Parameters:
    ----------------
    target: the groud truth , np.array
    prediction: the prediction, np.array
    '''

    return np.sum((target - prediction) ** 2) / np.sum((target ** 2))


def get_sae(target, prediction, sample_second):
    '''
    compute the signal aggregate error
    sae = |\hat(r)-r|/r where r is the ground truth total energy;
    \hat(r) is the predicted total energy.
    '''
    r = np.sum(target * sample_second * 1.0 / 3600.0)
    rhat = np.sum(prediction * sample_second * 1.0 / 3600.0)

    sae = np.abs(r - rhat) / np.abs(r)

    return sae

def get_Epd(target, prediction, sample_second):
    '''
    Energy per day
    - calculate energy of a day for both ground truth and prediction
    - sum all the energies
    - divide by the number of days
    '''

    day = int(24.0 * 3600 / sample_second)
    gt_en_days = []
    pred_en_days = []

    for start in range(0, int(len(target)-day), int(day)):
        gt_en_days.append(np.sum(target[start:start+day]*sample_second)/3600)
        pred_en_days.append(np.sum(prediction[start:start+day]*sample_second)/3600)

    Epd = np.sum(np.abs(np.array(gt_en_days)-np.array(pred_en_days)))/(len(target)/day)

    return Epd


def get_statistics(data):

    mean = np.mean(data)
    std = np.std(data)
    min_v = np.sort(data)[0]
    max_v = np.sort(data)[-1]

    quartile1 = np.percentile(data, 25)
    median = np.percentile(data, 50)
    quartile2 = np.percentile(data, 75)

    return mean, std, min_v, max_v, quartile1, median, quartile2


#####################################################################
def tp_tn_fp_fn(states_pred, states_ground):
    tp = np.sum(np.logical_and(states_pred == 1, states_ground == 1))
    fp = np.sum(np.logical_and(states_pred == 1, states_ground == 0))
    fn = np.sum(np.logical_and(states_pred == 0, states_ground == 1))
    tn = np.sum(np.logical_and(states_pred == 0, states_ground == 0))
    return tp, tn, fp, fn

def recall_precision_accuracy_f1(pred, ground,threshold):
    # aligned_meters = align_two_meters(pred, ground)
    # data = {
    #     'pred': pred,
    #     'truth': ground
    # }
    # df = pd.DataFrame(data)
    threshold = threshold
    chunk_results = []

    sum_samples = len(pred)
    pr = np.array([0 if (p)<threshold else 1 for p in pred])
    gr = np.array([0 if p<threshold else 1 for p in ground])

    tp, tn, fp, fn = tp_tn_fp_fn(pr,gr)
    p = sum(pr)
    n = len(pr) - p

    chunk_results.append([tp,tn,fp,fn,p,n])

    if sum_samples == 0:
        return None
    else:
        # [tp,tn,fp,fn,p,n] = np.sum(chunk_results, axis=0)

        res_recall = recall(tp,fn)
        res_precision = precision(tp,fp)
        res_f1 = f1(res_precision,res_recall)
        res_accuracy = accuracy(tp,tn,p,n)

        return (res_recall,res_precision,res_accuracy,res_f1)

def relative_error_total_energy(pred, ground):
    aligned_meters = align_two_meters(pred, ground)
    chunk_results = []
    sum_samples = 0.0
    for chunk in aligned_meters:
        chunk.fillna(0, inplace=True)
        sum_samples += len(chunk)
        E_pred = sum(chunk.iloc[:,0])
        E_ground = sum(chunk.iloc[:,1])

        chunk_results.append([
                            E_pred,
                            E_ground
                            ])
    if sum_samples == 0:
        return None
    else:
        [E_pred, E_ground] = np.sum(chunk_results,axis=0)
        return abs(E_pred - E_ground) / float(max(E_pred,E_ground))

def mean_absolute_error(pred, ground):
    aligned_meters = align_two_meters(pred, ground)
    total_sum = 0.0
    sum_samples = 0.0
    for chunk in aligned_meters:
        chunk.fillna(0, inplace=True)
        sum_samples += len(chunk)
        total_sum += sum(abs((chunk.iloc[:,0]) - chunk.iloc[:,1]))
    if sum_samples == 0:
        return None
    else:
        return total_sum / sum_samples


def recall(tp,fn):
    return tp/float(tp+fn)

def precision(tp,fp):
    return tp/float(tp+fp)

def f1(prec,rec):
    return 2 * (prec*rec) / float(prec+rec)

def accuracy(tp, tn, p, n):
    return (tp + tn) / float(p + n)