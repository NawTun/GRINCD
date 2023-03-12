import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats


def dream5():
    """
    Top k accuracy rate of methods on the DREAM5 datasets.
    """
    top_k = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000]
    methods = os.listdir("./output/DREAM5_alg_output")
    our_in_silico_res = []
    our_e_coli_res = []
    our_s_cere_res = []
    for i in tqdm(range(100)):
        tmp_data1 = pd.read_csv("./output/in_silico/prediction_" + str(i + 1) + ".txt", sep="\t", header=None)
        tmp_data2 = pd.read_csv("./output/e_coli/prediction_" + str(i + 1) + ".txt", sep="\t", header=None)
        tmp_data3 = pd.read_csv("./output/s_cere/prediction_" + str(i + 1) + ".txt", sep="\t", header=None)
        our_in_silico_res.append(tmp_data1)
        our_e_coli_res.append(tmp_data2)
        our_s_cere_res.append(tmp_data3)
    for k in tqdm(top_k):
        cross_table = pd.DataFrame(columns=methods, index=["in_silico", "e_coli", "s_cere"])
        for m in methods:
            print(m)
            in_silico_res = pd.read_csv("./output/DREAM5_alg_output/" + m + "/in_silico.txt", sep="\t", header=None)
            e_coli_res = pd.read_csv("./output/DREAM5_alg_output/" + m + "/e_coli.txt", sep="\t", header=None)
            s_cere_res = pd.read_csv("./output/DREAM5_alg_output/" + m + "/s_cere.txt", sep="\t", header=None)
            in_silico_cross = []
            e_coli_cross = []
            s_cere_cross = []
            for i in range(100):
                in_silico_cross.append(pd.merge(in_silico_res.iloc[0:k, :], our_in_silico_res[i].iloc[0:k, :]).shape[0])
                e_coli_cross.append(pd.merge(e_coli_res.iloc[0:k, :], our_e_coli_res[i].iloc[0:k, :]).shape[0])
                s_cere_cross.append(pd.merge(s_cere_res.iloc[0:k, :], our_s_cere_res[i].iloc[0:k, :]).shape[0])
            # print("in_silico:", np.average(in_silico_cross))
            # print("e_coli:", np.average(e_coli_cross))
            # print("s_cere:", np.average(s_cere_cross))
            cross_table.loc["in_silico", m] = np.average(in_silico_cross)
            cross_table.loc["e_coli", m] = np.average(e_coli_cross)
            cross_table.loc["s_cere", m] = np.average(s_cere_cross)
        cross_table.to_csv("k=" + str(k) + ".csv", sep=",")


def single_cell():
    """
    Top k accuracy rate of methods on the single cell datasets.
    """
    top_k = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000]
    methods = os.listdir("./output/sc_alg_output")
    dataset_name = []
    cells = ["hESC", "mDC"]
    gs_type = ["Cell_type_specific", "Non_specific", "STRING"]
    gene_no = ["500", "1000"]
    for cell in cells:
        for gs in gs_type:
            for no in gene_no:
                dataset_name.append(cell + "_" + gs + "_" + no)
    for k in tqdm(top_k):
        cross_table = pd.DataFrame(columns=methods, index=dataset_name)
        for cell in cells:
            for gs in gs_type:
                for no in gene_no:
                    print(cell + "/" + gs + "/" + no)
                    our_res = []
                    for i in range(100):
                        tmp_data = pd.read_csv(
                            "./output/" + cell + "/" + gs + "/" + no + "/prediction_" + str(i + 1) + ".txt", sep="\t",
                            header=None)
                        our_res.append(tmp_data)
                    for m in methods:
                        # print(m)
                        res = pd.read_csv("./output/sc_alg_output/" + m + "/" + cell + "_" + gs + "_" + no + ".txt",
                                          sep="\t",
                                          header=None)
                        cross = []
                        for j in range(100):
                            cross.append(
                                pd.merge(res.iloc[0:k, :], our_res[j].iloc[0:k, :]).shape[0])
                        cross_table.loc[cell + "_" + gs + "_" + no, m] = np.average(cross)
        cross_table.to_csv("k=" + str(k) + ".csv", sep=",")
        print(cross_table)


def demo():
    data = pd.read_csv(
        "/Users/lgge/Library/CloudStorage/OneDrive-个人/论文/res_for_chart/colon_cancer/raw_data_and_exp_data/GSE4183.csv",
        sep=",", header=0, index_col=0)
    normal = ["GSM95473",
              "GSM95474",
              "GSM95475",
              "GSM95476",
              "GSM95477",
              "GSM95478",
              "GSM95479",
              "GSM95480", ]
    cancer = [
        "GSM95496",
        "GSM95497",
        "GSM95498",
        "GSM95499",
        "GSM95500",
        "GSM95501",
        "GSM95502",
        "GSM95503",
        "GSM95504",
        "GSM95505",
        "GSM95506",
        "GSM95507",
        "GSM95508",
        "GSM95509",
        "GSM95510", ]
    tf_candidate, tg_candidate = exp0625()
    normal_sample = data.loc[:, normal]
    cancer_sample = data.loc[:, cancer]
    res = dict()
    for row in tqdm(data.index):
        tmp_res = stats.ttest_ind(normal_sample.loc[row, :], cancer_sample.loc[row, :], equal_var=False)[1]
        # print(row)
        # print(tmp_res)
        res[row] = tmp_res
    final_tf = [k for k, v in res.items() if v < 0.01 and k in tf_candidate]
    print("final_tf:")
    print(final_tf)
    print(len(final_tf))
    res = sorted(res.items(), key=lambda x: x[1], reverse=False)
    final_tg = [k[0] for k in res if k[1] < 0.01 and k[0] in tg_candidate][0:4 * len(final_tf)]
    # print("final_tg:")
    # print(final_tg)
    # print(len(final_tg))
    final_gene = final_tf + final_tg
    final_data = data.loc[final_gene, :].T
    normal_exp = final_data.iloc[0:8, :]
    adenoma_exp = final_data.iloc[8:23, :]
    CRC_exp = final_data.iloc[23:38, :]
    IBD_exp = final_data.iloc[38:, :]
    data_dict = {
        "normal": normal_exp,
        "adenoma": adenoma_exp,
        "CRC": CRC_exp,
        "IBD": IBD_exp
    }
    print(data_dict)
    print(data_dict["normal"].columns)
    for k, v in data_dict.items():
        v.to_csv(k + ".txt", sep="\t", header=True, index=False)


def exp0625():
    """
    Screening for specific expression genes in the colon.
    """
    human_tfs_df = pd.read_csv(
        "/Users/lgge/Library/CloudStorage/OneDrive-个人/论文/res_for_chart/colon_cancer/exp0625/tf_human.txt",
        sep="\t", header=None)
    human_tfs = set(human_tfs_df.iloc[:, 0])
    tissue_spec_genes_df = pd.read_csv(
        "/Users/lgge/Library/CloudStorage/OneDrive-个人/论文/res_for_chart/colon_cancer/exp0625/tissue_spec_genes.txt",
        header=None, sep="\t")
    tissue_spec_genes_df = tissue_spec_genes_df.loc[tissue_spec_genes_df[2] != "Colon", :]
    other_tis_spec_genes = set(tissue_spec_genes_df[0])
    # print(other_tis_spec_genes)
    exp_data = pd.read_csv(
        "/Users/lgge/Library/CloudStorage/OneDrive-个人/论文/res_for_chart/colon_cancer/raw_data_and_exp_data/GSE4183.csv",
        header=0, index_col=0)
    # print(exp_data)
    exp_data_genes = set(exp_data.index)
    # print(human_tfs)
    # print("original human tfs numbers:", len(human_tfs))
    candidate_tfs = human_tfs - other_tis_spec_genes
    # print(candidate_tfs)
    # print("candidate tfs number:", len(candidate_tfs))
    candidate_genes = exp_data_genes - other_tis_spec_genes
    # print("candidate genes number:", len(candidate_genes))
    tfs = candidate_tfs & candidate_genes
    tgs = candidate_genes - tfs
    # print(len(tfs))
    # print(len(tgs))
    return tfs, tgs





if __name__ == '__main__':
    # dream5()
    # single_cell()
    demo()
    # pass
