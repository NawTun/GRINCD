import pandas as pd
from sklearn.utils import shuffle
import numpy as np
import scipy.io as scio
from tqdm.auto import tqdm
from numpy import mean


def edgelist_2_D5C4_gs(gold_positives):
    TFs = np.unique(gold_positives[0].values)
    targets = np.unique(gold_positives[[0, 1]].values)
    max_TF = max(TFs)
    max_target = max(targets)
    G = np.ones((max_TF, max_target), dtype=np.int) * -1
    missed_TFs = list(set(range(1, max_TF + 1)).difference(set(TFs)))
    missed_targets = list(set(range(1, max_target + 1)).difference(set(targets)))
    missed_indexes = np.array(missed_TFs) - 1
    missed_cols = np.array(missed_targets) - 1
    if len(missed_indexes) != 0:
        G[missed_indexes, :] = 0
    if len(missed_cols) != 0:
        G[:, missed_cols] = 0
    np.fill_diagonal(G, 0)
    G[(gold_positives[0] - 1).tolist(), (gold_positives[1] - 1).tolist()] = 1
    return G


def remove_edges_not_in_gs(prediction, G):
    TFs, targets = G.shape
    prediction = prediction[(prediction[0] <= TFs) & (prediction[1] <= targets)]
    i = prediction[0] - 1
    j = prediction[1] - 1
    tmp = G[i, j] != 0
    prediction_cleaned = prediction[tmp]
    return prediction_cleaned


def probability(X, Y, x):
    X = X.squeeze(0)
    Y = Y.squeeze(0)
    dx = X[1] - X[0]
    tmp = (X >= x).astype(np.float)
    P = np.sum(tmp * Y * dx)
    return P


def evaluation(gold_positives, prediction_raw, pdf_aupr, pdf_auroc):
    G = edgelist_2_D5C4_gs(gold_positives)
    P = np.sum(G == 1)
    N = np.sum(G == -1)
    T = P + N
    prediction = remove_edges_not_in_gs(prediction_raw, G)
    L = prediction.shape[0]
    discovery = np.zeros((L, 1), dtype=np.int)
    discovery[G[prediction[0] - 1, prediction[1] - 1] == 1] = 1
    TPL = np.sum(discovery)
    if L < T:
        p = (P - TPL) / (T - L)
    else:
        p = 0
    random_positive_discovery = np.ones((T - L, 1))
    random_negative_discovery = np.ones((T - L, 1))
    random_positive_discovery = random_positive_discovery * p
    random_negative_discovery = random_negative_discovery * (1 - p)
    if 0 in random_positive_discovery.shape:
        positive_discovery = discovery
    else:
        positive_discovery = np.vstack((discovery, random_positive_discovery))
    if 0 in random_negative_discovery.shape:
        negative_discovery = 1 - discovery
    else:
        negative_discovery = np.vstack((1 - discovery, random_negative_discovery))
    TPk = np.cumsum(positive_discovery)
    FPk = np.cumsum(negative_discovery)
    K = np.arange(1, (T + 1))
    TPR = TPk / P
    FPR = FPk / N
    REC = TPR
    PREC = TPk / K
    if ((P != round(TPk[-1])) | (N != round(FPk[-1]))):
        print('ERROR. There is a problem with the completion of the prediction list.')
    TPk[-1] = round(TPk[-1])
    FPk[-1] = round(FPk[-1])
    AUROC = np.trapz(TPR, FPR)
    AUPR = np.trapz(PREC, REC) / (1 - 1 / P)
    X = pdf_aupr['X']
    Y = pdf_aupr['Y']
    P_AUPR = probability(X, Y, AUPR)
    X = pdf_auroc['X']
    Y = pdf_auroc['Y']
    P_AUROC = probability(X, Y, AUROC)
    return TPR, FPR, PREC, REC, L, AUROC, AUPR, P_AUROC, P_AUPR


def EP(gold_standard, prediction):
    k = gold_standard.shape[0]
    gs_set = set(zip(gold_standard[0], gold_standard[1]))
    pred = prediction.iloc[:k, :]
    pred_set = set(zip(pred[0], pred[1]))
    l = len(gs_set.intersection(pred_set))
    precision = l / k
    return precision


def EPR(gold_standard, prediction):
    ep = EP(gold_standard, prediction)
    random_ep = []
    for i in range(100):
        random_net = shuffle(prediction)
        random_ep.append(EP(gold_standard, random_net))
    epr = ep / np.mean(random_ep)
    return epr


def run(prediction, gold_standard, pdf_aupr, pdf_auroc):
    gold_standard[0] = gold_standard[0].str[1:].astype(int)
    gold_standard[1] = gold_standard[1].str[1:].astype(int)
    prediction.columns = [0, 1]
    prediction[0] = prediction[0].str[1:].astype(int)
    prediction[1] = prediction[1].str[1:].astype(int)
    epr = EPR(gold_standard, prediction)
    tpr, fpr, prec, rec, l, auroc, aupr, p_auroc, p_aupr = evaluation(gold_standard, prediction, pdf_aupr, pdf_auroc)
    if p_aupr == 0:
        aupr_score = 100
    else:
        aupr_score = 100 if -np.log10(p_aupr) > 100 else -np.log10(p_aupr)
    if p_auroc == 0:
        auroc_score = 100
    else:
        auroc_score = 100 if -np.log10(p_auroc) > 100 else -np.log10(p_auroc)
    conf_score = (aupr_score + auroc_score) / 2
    return auroc, aupr, conf_score, epr


def main():
    dataset_list = ["in_silico", "e_coli", "s_cere", "hESC", "mDC"]
    cell_type_list = ["Cell_type_specific", "Non_specific", "STRING"]
    num_list = ["500", "1000"]
    final_dataset_list = []
    for dataset in dataset_list:
        if dataset in ["hESC", "mDC"]:
            for cell_type in cell_type_list:
                for num in num_list:
                    final_dataset_list.append(dataset + "/" + cell_type + "/" + num)
        else:
            final_dataset_list.append(dataset)
    for dataset in final_dataset_list:
        input_dir = "input/" + dataset + "/"
        output_dir = "output/" + dataset + "/"
        print("evaluating " + dataset + " ......")
        auroc_list = []
        aupr_list = []
        conf_score_list = []
        epr_list = []
        pdf_auroc = scio.loadmat(input_dir + "AUROC.mat")
        pdf_aupr = scio.loadmat(input_dir + "AUPR.mat")
        for i in tqdm(range(100)):
            gs = pd.read_csv(input_dir + "gold_standard.tsv", sep="\t", header=None)
            pred = pd.read_csv(output_dir + "prediction_" + str(i + 1) + ".txt", sep="\t", header=None)
            auroc, aupr, conf_score, epr = run(pred, gs, pdf_aupr, pdf_auroc)
            auroc_list.append(auroc)
            aupr_list.append(aupr)
            conf_score_list.append(conf_score)
            epr_list.append(epr)
        print("average results are: \n", "\tauroc:", mean(auroc_list), "\n\taupr:", mean(aupr_list), "\n\tconf_score:",
              mean(conf_score_list), "\n\tepr:", mean(epr_list))


if __name__ == '__main__':
    main()
