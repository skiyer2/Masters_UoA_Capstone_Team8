import numpy as np
import math
import torch
from sklearn.metrics import roc_auc_score

def get_auc(pre_y, y):
    pre_y = np.asarray(pre_y).flatten()
    y = np.asarray(y).flatten()
    mask = y != -1
    if mask.sum() == 0:
        return 0
    return roc_auc_score(y[mask], pre_y[mask])

def F1_and_recall(ranked_list, ground_list):
    hits = 0
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id in ground_list:
            hits += 1
    pre = hits/(1.0 * len(ranked_list)+1)
    rec = hits/(1.0 * len(ground_list)+1)
    if pre != 0 or rec != 0:
        f1 = (pre*rec*2)/(pre+rec)
    else:
        f1 = 0
    return f1, rec

def AP(ranked_list, ground_truth):
    hits = 0
    sum_precs = 0
    for n in range(len(ranked_list)):
        if ranked_list[n] in ground_truth:
            hits += 1
            sum_precs += hits / (n + 1.0)
    if hits > 0:
        return sum_precs / len(ranked_list)
    else:
        return 0

def IDCG(n):
    idcg = 0
    for i in range(n):
        idcg += 1 / math.log(i+2, 2)
    return idcg

def nDCG(ranked_list, ground_truth):
    dcg = 0
    idcg = IDCG(len(ranked_list))
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id not in ground_truth:
            continue
        rank = i+1
        dcg += 1 / math.log(rank+1, 2)
    return dcg / idcg
        
def MRR(ranked_list, ground_list):
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id in ground_list:
            return 1 / (i + 1.0)
    return 0

def topN(args, recommend_list_candidate, now_list, now_user):
    ap_list3 = []
    ndcg_list3 = []
    rr_list3 = []
    f1_list3 = []
    rec_list3 = []

    ap_list5 = []
    ndcg_list5 = []
    rr_list5 = []
    f1_list5 = []
    rec_list5 = []

    ap_list10 = []
    ndcg_list10 = []
    rr_list10 = []
    f1_list10 = []
    rec_list10 = []

    ret = {}

    for index, user in enumerate(now_user):
        recommend_list = []
        for i in recommend_list_candidate[index]:
            if i in args.train_usertoitems[user]:
                continue
            else:
                recommend_list.append(i)
            if len(recommend_list) == args.topN:
                break
        
        recommend_list = recommend_list_candidate[index]
        ALL_group_list = list(now_list[user])

        ap_list3.append(AP(recommend_list[:3], ALL_group_list))
        ndcg_list3.append(nDCG(recommend_list[:3], ALL_group_list))
        rr_list3.append(MRR(recommend_list[:3], ALL_group_list))
        f1, rec = F1_and_recall(recommend_list[:3], ALL_group_list)
        f1_list3.append(f1)
        rec_list3.append(rec)

        ap_list5.append(AP(recommend_list[:5], ALL_group_list))
        ndcg_list5.append(nDCG(recommend_list[:5], ALL_group_list))
        rr_list5.append(MRR(recommend_list[:5], ALL_group_list))
        f1, rec = F1_and_recall(recommend_list[:5], ALL_group_list)
        f1_list5.append(f1)
        rec_list5.append(rec)

        ap_list10.append(AP(recommend_list[:10], ALL_group_list))
        ndcg_list10.append(nDCG(recommend_list[:10], ALL_group_list))
        rr_list10.append(MRR(recommend_list[:10], ALL_group_list))
        f1, rec = F1_and_recall(recommend_list[:10], ALL_group_list)
        f1_list10.append(f1)
        rec_list10.append(rec)

    ret['ap_list3'] = ap_list3
    ret['ndcg_list3'] = ndcg_list3
    ret['rr_list3'] = rr_list3
    ret['f1_list3'] = f1_list3
    ret['rec_list3'] = rec_list3

    ret['ap_list5'] = ap_list5
    ret['ndcg_list5'] = ndcg_list5
    ret['rr_list5'] = rr_list5
    ret['f1_list5'] = f1_list5
    ret['rec_list5'] = rec_list5

    ret['ap_list10'] = ap_list10
    ret['ndcg_list10'] = ndcg_list10
    ret['rr_list10'] = rr_list10
    ret['f1_list10'] = f1_list10
    ret['rec_list10'] = rec_list10
    return ret

def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')

def RecallPrecision_ATk(test_data, r, k):
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = []
    del_index = []
    for i in range(len(test_data)):
        tmp = len(test_data[i])
        if tmp != 0:
            recall_n.append(tmp)
        else:
            del_index.append(i)
    right_pred = np.delete(right_pred, del_index, axis=0)
    recall_n = np.array(recall_n)
    recall = np.sum(right_pred/recall_n)
    precis = np.sum(right_pred)/precis_n
    return {'recall': recall, 'precision': precis}

def MRRatK_r(r, k):
    pred_data = r[:, :k]
    scores = np.log2(1./np.arange(1, k+1))
    pred_data = pred_data/scores
    pred_data = pred_data.sum(1)
    return np.sum(pred_data)

def NDCGatK_r(test_data, r, k):
    assert len(r) == len(test_data)
    pred_data = r[:, :k]
    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1./np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data*(1./np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg/idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)

def test_one_batch(X, args):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    ret = RecallPrecision_ATk(groundTrue, r, args.topN)
    pre.append(ret['precision'])
    recall.append(ret['recall'])
    ndcg.append(NDCGatK_r(groundTrue, r, args.topN))
    if np.isnan(ret['recall']):
        print(1)
    return {'recall': np.array(recall), 
            'precision': np.array(pre), 
            'ndcg': np.array(ndcg)}
