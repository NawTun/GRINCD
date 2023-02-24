from multiprocessing import Process
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import os
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import scipy.io as scio
import itertools
from sklearn.utils import shuffle
import eval

# calculate mutual information of two variables, X and Y are vectors, bins is the number of intervals used to discrete vectors.
def calc_MI(X, Y, bins):
    XY_hist = np.histogram2d(X, Y, bins)[0]
    X_hist = np.histogram(X, bins)[0]
    Y_hist = np.histogram(Y, bins)[0]
    X_entropy = calc_entropy(X_hist)
    Y_entropy = calc_entropy(Y_hist)
    XY_entropy = calc_entropy(XY_hist)
    MI = X_entropy + Y_entropy - XY_entropy
    return MI

# calculate entropy of vector c.
def calc_entropy(c):
    c_prob = c / float(np.sum(c))
    c_prob = c_prob[np.nonzero(c_prob)]
    H = -sum(c_prob * np.log2(c_prob))
    return H

# create non-linear co-expression network using mutual information, this calculation would skipped if network csv file already exist.
def mi_net(tf_num, data_file, threshold=0.3, ):
    expr = pd.read_csv(data_file + "/expression_data.tsv", sep='\t', header=0)
    expr = (expr - expr.mean()) / (expr.std())
    mi_net_file = data_file + "/mi.tsv"
    if os.path.exists(mi_net_file):
        mi_mtx = pd.read_csv(mi_net_file, sep="\t", header=None).values
    else:
        mi_mtx = np.zeros((expr.shape[1], expr.shape[1]))
        for tf in tqdm(range(tf_num)):
            for tg in range(expr.shape[1]):
                A = expr.iloc[:, tf].values
                B = expr.iloc[:, tg].values
                result_NMI = calc_MI(A, B, bins=10)
                mi_mtx[tf, tg] = result_NMI
        tmp = pd.DataFrame(mi_mtx)
        tmp.to_csv(mi_net_file, sep="\t", header=False, index=False)
    mi_mtx = np.triu(mi_mtx, 0)
    mi_mtx = mi_mtx + mi_mtx.T
    np.fill_diagonal(mi_mtx, 0)
    mi_mtx[np.where(abs(mi_mtx) > threshold)] = 1
    mi_mtx[np.where(abs(mi_mtx) <= threshold)] = 0
    mi_mtx = pd.DataFrame(mi_mtx, index=expr.columns, columns=expr.columns, dtype=int)
    print(mi_mtx)
    return mi_mtx


def get_dir_list():
    net_list = ["in_silico", "e_coli", "s_cere", "hESC", "mDC"]
    input_base_dir = "/home/fengke/pycode/my_exp/paper/exp/input/"
    input_dir_list = []
    for net in net_list:
        if net in ["in_silico", "e_coli", "s_cere"]:
            input_dir_list.append(input_base_dir + net)
        else:
            for type in ["Cell_type_specific", "Non_specific", "STRING"]:
                for num in ["500", "1000"]:
                    input_dir_list.append(input_base_dir + net + "/" + type + "/" + num)
    return input_dir_list


def main():
    input_dir_list = get_dir_list()
    process_list = []
    for i in input_dir_list:
        tfs = list(pd.read_csv(i + "/transcription_factors.tsv", sep="\t", header=None)[0])
        tf_num = len(tfs)
        p = Process(target=mi_net, args=(tf_num, i))
        p.start()
        process_list.append(p)

    for p in process_list:
        p.join()


# generate random network auroc and aupr distrubution
def generate_au_dist():
    input_dir_list = get_dir_list()
    dir_list = [i for i in input_dir_list if ('in_silico' not in i) and ('e_coli' not in i) and ('s_cere' not in i)]
    for dir in tqdm(dir_list):
        print(dir)
        plt.cla()
        plt.rcParams['figure.dpi'] = 300
        eval_res = dir + "/random_eval_res.tsv"
        res = pd.read_csv(eval_res, sep="\t", header=0)
        sns.set()
        auroc = res['auroc']
        aupr = res['aupr']
        auroc_fig = sns.histplot(auroc, binwidth=0.00001, kde=True, log_scale=(False, 10)).set_title(dir[43:])
        auroc_fig = auroc_fig.get_figure()
        auroc_fig.savefig(dir + '/auroc.png', dpi=300)
        X_plot = np.linspace(0, 1, 100000)
        scipy_kde = st.gaussian_kde(auroc)
        auroc_dense = scipy_kde(X_plot)
        plt.cla()
        # ------------------------------------
        aupr_fig = sns.histplot(aupr, binwidth=0.00001, kde=True, log_scale=(False, 10)).set_title(dir[43:])
        aupr_fig = aupr_fig.get_figure()
        aupr_fig.savefig(dir + '/aupr.png', dpi=300)
        scipy_kde = st.gaussian_kde(aupr)
        aupr_dense = scipy_kde(X_plot)
        scio.savemat(dir + "/AUROC.mat", {"X": X_plot, "Y": auroc_dense})
        scio.savemat(dir + "/AUPR.mat", {"X": X_plot, "Y": aupr_dense})


# generate random networks
def gen_random_network(net_name, gs_type, num):
    base_dir = "./input/"
    tfs = pd.read_csv(base_dir + net_name + "/transcription_factors.tsv", sep="\t", header=None)[0]
    genes = pd.read_csv(base_dir + net_name + "/gene_ids.tsv", sep="\t", header=0)["#ID"]
    gs = pd.read_csv(base_dir + net_name + "/gold_standard.tsv", sep="\t", header=None)
    pdf_auroc = scio.loadmat(base_dir + net_name + "/AUROC.mat")
    pdf_aupr = scio.loadmat(base_dir + net_name + "/AUPR.mat")
    new = itertools.product(tfs, genes)
    new = pd.DataFrame(new)
    new = new.drop(new[(new[0] == new[1])].index)
    print(new)
    auroc_list = []
    aupr_list = []
    for _ in tqdm(range(num)):
        random_edge_list = shuffle(new)
        auroc, aupr, conf_score = eval.run(prediction=random_edge_list, gold_standard=gs, pdf_auroc=pdf_auroc,
                                           pdf_aupr=pdf_aupr)
        print(auroc)
        print(aupr)
        auroc_list.append(auroc)
        aupr_list.append(aupr)
    ret = pd.DataFrame(columns=["auroc", "aupr"])
    ret["auroc"] = auroc_list
    ret["aupr"] = aupr_list
    print(ret)
    ret.to_csv(net_name + "_" + gs_type + ".tsv", sep="\t", header=True, index=False)

# generate a ranked list for a weighted matrix W.
def generate_res(W, tf_num, genes):
    W = np.abs(W)
    np.fill_diagonal(W, 0)
    W_prime = W[range(tf_num), :]
    idx, col = np.where(abs(W_prime) > 0)
    print("-------------")
    # edges_df = pd.DataFrame(
    #     {'TF': genes[idx], 'Target': genes[col], 'Confidence': (W_prime[idx, col])})
    edges_df = pd.DataFrame(
        {'TF': genes[idx], 'Target': genes[col], 'Confidence': (W_prime[idx, col])})
    edges_df = edges_df.sort_values('Confidence', ascending=False)
    return edges_df
