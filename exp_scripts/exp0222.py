import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
from arboreto.algo import grnboost2
import matplotlib


def judge_monotonic(x, threshold):
    up, down = 0, 0
    zero = 0
    A = [x["normal"], x["IBD"], x["adenoma"], x["CRC"]]
    for i in range(3):
        if threshold <= (A[i + 1] - A[i]):
            up = 1
        elif (A[i] - A[i + 1]) >= threshold:
            down = 1
        else:
            zero += 1
    if up + down == 1 and zero <= 1:
        if up == 1:
            return "up"
        else:
            return "down"
    else:
        return "no"


def exp0107(threshold):
    # find key factors for colon cancer using monotonicity
    situation_dict = {"normal": set(), "IBD": set(), "adenoma": set(), "CRC": set()}
    all_data = pd.DataFrame(columns=["tf", "tg", "normal", "IBD", "adenoma", "CRC"])
    for i in situation_dict.keys():
        data = pd.read_csv("/Users/lgge/Desktop/colon_cancer_tigress/" + i + ".txt",
                           sep=",", header=None)
        data.columns = ["tf", "tg", "val"]
        data['val'] = data['val'] * 2
        data = data.sort_values(by=["tf", "tg"], ascending=(True, True))
        all_data[i] = data["val"].tolist()
        all_data["tf"] = data["tf"].tolist()
        all_data["tg"] = data["tg"].tolist()

    # all_data["sub1"] = all_data["IBD"] - all_data["normal"]
    # all_data["sub2"] = all_data["adenoma"] - all_data["IBD"]
    # all_data["sub3"] = all_data["CRC"] - all_data["adenoma"]
    all_data["monotonic"] = "no"
    # print(all_data)
    all_data['monotonic'] = all_data.apply(judge_monotonic, args=(threshold,), axis=1)
    # print(all_data[all_data["monotonic"] == True])
    # print(all_data[all_data["monotonic"] == True].shape)
    # print(len(set(all_data[all_data["monotonic"] == True]["tf"])))
    extracted_up_edges = all_data[all_data["monotonic"] == "up"]
    extracted_down_edges = all_data[all_data["monotonic"] == "down"]
    print(extracted_up_edges)
    print(extracted_down_edges)
    print(extracted_up_edges.shape)
    print(extracted_down_edges.shape)
    final_tf_set = set(extracted_up_edges['tf']) | set(extracted_down_edges['tf'])
    print(len(final_tf_set))
    print(final_tf_set)
    # extracted_edges = extracted_edges.loc[:, ["tf", "tg"]]
    # print(extracted_edges)
    # extracted_edges.to_csv("/Users/lgge/Desktop/edges.txt", sep="\t", header=False)
    # print("num of extracted edges:", extracted_edges.shape[0])
    # print(extracted_edges["tf"].value_counts())
    # return set(all_data[all_data["monotonic"] == True]["tf"])


def run_grnboost2(threshold):
    f = open("/Users/lgge/Desktop/colon_data/tgs.txt")
    tgs = []
    tfs = []
    while True:
        line = f.readline()
        if (line != ""):
            tgs.append(line.rstrip("\n"))
        else:
            break
    f.close()
    f = open("/Users/lgge/Desktop/colon_data/tfs.txt")
    while True:
        line = f.readline()
        if (line != ""):
            tfs.append(line.rstrip("\n"))
        else:
            break
    f.close()
    type_list = ["normal", "IBD", "adenoma", "CRC"]
    situation_dict = {"normal": set(), "IBD": set(), "adenoma": set(), "CRC": set()}
    all_data = pd.DataFrame(columns=["tf", "tg", "normal", "IBD", "adenoma", "CRC"])
    for i in type_list:
        ex_matrix = pd.read_csv("/Users/lgge/Desktop/test/" + i + ".txt", sep="\t", header=0)
        data = grnboost2(expression_data=ex_matrix, gene_names=list(ex_matrix.columns), tf_names=tfs)
        print(data.shape)
        data.columns = ["tf", "tg", "val"]
        data = data.sort_values(by=["tf", "tg"], ascending=(True, True))
        all_data[i] = data["val"].tolist()
        all_data["tf"] = data["tf"].tolist()
        all_data["tg"] = data["tg"].tolist()
    all_data["monotonic"] = "no"
    # print(all_data)
    all_data['monotonic'] = all_data.apply(judge_monotonic, args=(threshold,), axis=1)
    # print(all_data[all_data["monotonic"] == True])
    # print(all_data[all_data["monotonic"] == True].shape)
    # print(len(set(all_data[all_data["monotonic"] == True]["tf"])))
    extracted_up_edges = all_data[all_data["monotonic"] == "up"]
    extracted_down_edges = all_data[all_data["monotonic"] == "down"]
    print(extracted_up_edges)
    print(extracted_down_edges)
    print(extracted_up_edges.shape)
    print(extracted_down_edges.shape)
    final_tf_set = set(extracted_up_edges['tf']) | set(extracted_down_edges['tf'])
    print(len(final_tf_set))
    print(final_tf_set)


def draw_plot():
    base_dir = "/Users/lgge/Desktop/单细胞数据交集"
    print(os.listdir(base_dir))
    df_dict = dict()
    for k in os.listdir(base_dir):
        df_dict[k] = pd.read_csv(base_dir + "/" + k, sep=",", header=0, index_col=0)
        # print(df_dict[k])
    in_silico_df = pd.DataFrame(
        columns=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000,
                 4000], index=list(df_dict["k=30.csv"].columns))
    for k, v in df_dict.items():
        print(k)
        for method in list(df_dict["k=30.csv"].columns):
            in_silico_df.loc[method, int(k.lstrip("k=").rstrip(".csv"))] = v.loc["hESC_STRING_1000", method]
    print(in_silico_df.T)
    # sns.palplot(sns.color_palette("Accent"))
    plt.figure(figsize=(10, 6))
    sns.lineplot(in_silico_df.T, palette="tab20_r")
    plt.show()


if __name__ == '__main__':
    # exp0107(0.22)
    # venn3([set(['WWTR1', 'E2F6', 'ZBTB7A', 'RUVBL1', 'URI1']),
    #        set(['CREM', 'NOTCH3', 'ZBTB7A', 'NR3C2', 'NR2F6', 'WWTR1']),
    #        set(['CEBPB', 'TFAP2C', 'ZNF24', 'ZBTB7A', 'NOTCH3', 'HDAC7', 'NFYC', 'TRRAP'])])
    # plt.show()
    # run_grnboost2(0.15)
    draw_plot()
