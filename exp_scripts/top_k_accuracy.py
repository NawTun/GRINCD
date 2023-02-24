import os
import numpy as np
import pandas as pd


def dream5():
    # 所有的gs数据
    methods_list = ["GRINCD"]
    for method in os.listdir("./output/DREAM5_alg_output"):
        if method.startswith("."):
            continue
        methods_list.append(method)
    data_list = ["in_silico", "e_coli", "s_cere"]
    gs_dict = dict()
    gs_base_dir = "./input/"
    for data in data_list:
        gs_dict[data] = pd.read_csv(gs_base_dir + data + "/gold_standard.tsv", sep="\t", header=None)
    k_list = [50, 100, 200, 500, 1000, 2000, 3000, 4000]
    res_df = dict()

    for data in data_list:
        res_df[data] = pd.DataFrame(columns=k_list, index=methods_list)
        for k in k_list:
            for method in methods_list:
                if method.startswith("."):
                    continue
                if method == "GRINCD":
                    continue
                method_res = pd.read_csv("./output/DREAM5_alg_output/" + method + "/" + data + ".txt", sep="\t",
                                         header=None)
                res_df[data].loc[method][k] = pd.merge(method_res.iloc[:k, :], gs_dict[data], how="inner").shape[0] / k
            tmp_res = []
            for i in range(100):
                method_res = pd.read_csv("./output/" + data + "/prediction_" + str(i + 1) + ".txt", sep="\t",
                                         header=None)
                tmp_res.append(pd.merge(method_res.iloc[:k, :], gs_dict[data], how="inner").shape[0] / k)
            res_df[data].loc["GRINCD"][k] = np.average(tmp_res)

        print(res_df[data])
        res_df[data].to_csv(data + ".csv", sep=",", header=True, index=True)


def single_cell():
    # 所有的gs数据
    sc_data_list = []
    for i in ["hESC_", "mDC_"]:
        for j in ["Cell_type_specific", "Non_specific", "STRING"]:
            for k in ["_500", "_1000"]:
                sc_data_list.append(i + j + k)
    print(sc_data_list)
    methods_list = ["GRINCD"]
    for method in os.listdir("./output/sc_alg_output"):
        if method.startswith("."):
            continue
        methods_list.append(method)
    gs_dict = dict()
    gs_base_dir = "./input/"
    for data in sc_data_list:
        data_path = data.replace("_", "/", 1)
        data_path = data_path[::-1]
        data_path = data_path.replace("_", "/", 1)
        data_path = data_path[::-1]
        gs_dict[data] = pd.read_csv(gs_base_dir + data_path + "/gold_standard.tsv", sep="\t", header=None)
    k_list = [50, 100, 200, 500, 1000, 2000, 3000, 4000]
    res_df = dict()

    for data in sc_data_list:
        res_df[data] = pd.DataFrame(columns=k_list, index=methods_list)
        for k in k_list:
            for method in methods_list:
                if method.startswith("."):
                    continue
                if method == "GRINCD":
                    continue
                method_res = pd.read_csv("./output/sc_alg_output/" + method + "/" + data + ".txt", sep="\t",
                                         header=None)
                res_df[data].loc[method][k] = pd.merge(method_res.iloc[:k, :], gs_dict[data], how="inner").shape[0] / k
            tmp_res = []
            for i in range(100):
                data_path = data.replace("_", "/", 1)
                data_path = data_path[::-1]
                data_path = data_path.replace("_", "/", 1)
                data_path = data_path[::-1]
                method_res = pd.read_csv("./output/" + data_path + "/prediction_" + str(i + 1) + ".txt", sep="\t",
                                         header=None)
                tmp_res.append(pd.merge(method_res.iloc[:k, :], gs_dict[data], how="inner").shape[0] / k)
            res_df[data].loc["GRINCD"][k] = np.average(tmp_res)
        print(data)
        print(res_df[data])
        res_df[data].to_csv(data + ".csv", sep=",", header=True, index=True)


if __name__ == '__main__':
    # dream5()
    single_cell()
