import pandas as pd
from tqdm.auto import tqdm
import utils
import scipy.io as scio
import os
import eval
import numpy as np
from arboreto.utils import load_tf_names
from arboreto.algo import grnboost2
from multiprocessing import Process
from pandas import ExcelWriter
import matplotlib.pyplot as plt
import seaborn as sns
from anm import ANM


def padding(array, threshold):
    """
    Fine-tune AUROC and AUPR to calculate the appropriate score.
    :param array: AUROC or AUPR list.
    :param threshold: Threshold.
    :return: AUROC or AUPR list after fine-tuning.
    """
    i = 0
    pre_idx = 0
    post_idx = len(array) - 1
    while i < len(array):
        if array[pre_idx] < threshold:
            pre_idx += 1
        elif array[post_idx] < threshold:
            post_idx -= 1
        else:
            break
        i += 1
    pre_val = array[pre_idx]
    post_val = array[post_idx]
    print(pre_idx)
    print(post_idx)
    print(pre_val)
    print(post_val)
    for j in range(pre_idx):
        array[j] = (pre_val / pre_idx) * j
    for k in range(post_idx, len(array)):
        array[k] = (post_val / (len(array) - 1 - post_idx)) * (len(array) - 1 - k)
    return array


# 使用padding方法对pdf进行微调
def tune_pdf():
    """
    Use the padding method to fine-tune the pdf.
    """
    dir_list = utils.get_dir_list()[3:]
    print(len(dir_list))
    for i in dir_list:
        AUPR = scio.loadmat(i + "/AUPR.mat")
        AUROC = scio.loadmat(i + "/AUROC.mat")
        aupr_y = padding(AUPR["Y"][0], 1e-20)
        auroc_y = padding(AUROC["Y"][0], 1e-20)
        AUPR["Y"] = aupr_y
        AUROC["Y"] = auroc_y
        scio.savemat(i + "/AUPR.mat", AUPR)
        scio.savemat(i + "/AUROC.mat", AUROC)
        print('end')


def add_res():
    """
    Record the results of the evaluation to a file for statistics.
    """
    input_dirs = utils.get_dir_list()
    output_dir = [i.replace('input', 'output') for i in input_dirs[:3]]
    print(output_dir)
    for dir in output_dir:
        print(dir)
        gs = pd.read_csv(dir.replace('output', 'input') + "/gold_standard.tsv", sep="\t", header=None)
        pdf_aupr = scio.loadmat(dir.replace('output', 'input') + "/AUPR.mat")
        pdf_auroc = scio.loadmat(dir.replace('output', 'input') + "/AUROC.mat")
        files = os.listdir(dir)
        dir_files = [dir + "/" + file for file in files]
        print(dir_files)
        for file in tqdm(dir_files):
            idx = file.split("_")[-1]
            pred = pd.read_csv(file, sep="\t", header=None)
            auroc, aupr, conf_score, epr = eval.run(prediction=pred, gold_standard=gs.copy(), pdf_auroc=pdf_auroc,
                                                    pdf_aupr=pdf_aupr)
            print(auroc, aupr, conf_score, epr)
            with open(dir + '/res_' + idx, 'w+') as f:
                f.write(str(auroc))
                f.write('\n')
                f.write(str(aupr))
                f.write('\n')
                f.write(str(conf_score))
                f.write('\n')
                f.write(str(epr))


def collect_res():
    """
    Statistically evaluate the results and calculate the mean.
    """
    input_dirs = utils.get_dir_list()
    output_dir = [i.replace('input', 'output') for i in input_dirs[:3]]
    for dir in output_dir:
        print(dir)
        res_list = [i for i in os.listdir(dir) if "res" in i]
        auroc_list = []
        aupr_list = []
        score_list = []
        epr_list = []
        for res in res_list:
            with open(dir + "/" + res, 'r+') as f:
                auroc_list.append(float(f.readline()))
                aupr_list.append(float(f.readline()))
                score_list.append(float(f.readline()))
                epr_list.append(float(f.readline()))
        print(np.mean(auroc_list))
        print(np.mean(aupr_list))
        print(np.mean(score_list))
        print(np.mean(epr_list))


# def teams_score():
#     home_dir = "Challenge_participants"
#     net_mapping = {
#         "Network1": "in_silico",
#         "Network2": "",
#         "Network3": "e_coli",
#         "Network4": "s_cere",
#     }
#     teams_list = os.listdir(home_dir)
#     for team in teams_list:
#         print(team)
#         prediction_files = os.listdir(home_dir + "/" + team)
#         for file in prediction_files:
#             net_no = file.split("_")[3][:8]
#             net_name = net_mapping[file.split("_")[3][:8]]
#             if net_name != "":
#                 gs = pd.read_csv("input/" + net_name + "/gold_standard.tsv", sep="\t", header=None)
#                 pdf_auroc = scio.loadmat("input/" + net_name + "/AUROC.mat")
#                 pdf_aupr = scio.loadmat("input/" + net_name + "/AUPR.mat")
#                 pred = pd.read_csv(home_dir + "/" + team + "/" + file, sep="\t", header=None).iloc[:, :2]
#                 auroc, aupr, conf_score, epr = eval.run(prediction=pred, gold_standard=gs.copy(), pdf_auroc=pdf_auroc,
#                                                         pdf_aupr=pdf_aupr)
#                 print(net_name, auroc, aupr, conf_score, epr)
#         print("----------------------------")


def get_beeline_input():
    """
    Get the input of the Beeline dataset.
    """
    original_dir = utils.get_dir_list()[:3]
    print(original_dir)
    for dir in original_dir:
        exp_data = pd.read_csv(dir + "/expression_data.tsv", sep="\t", header=0).T
        print(exp_data)
        beeline_dir_name = "beeline_input/" + dir.split("/")[-1]
        print(beeline_dir_name)
        os.makedirs(beeline_dir_name)
        exp_data.to_csv(beeline_dir_name + "/ExpressionData.csv", sep=",", header=True, index=True)


def run_grnboost2():
    """
    Run GRNBOOST2 to infer GRN.
    """
    original_dir = utils.get_dir_list()[:3]
    for i in original_dir:
        print(i.split("/")[-1])
        ex_matrix = pd.read_csv(i + "/expression_data.tsv", sep='\t', header=0)
        gene_name = list(ex_matrix.columns)
        # tf_names is read using a utility function included in Arboreto
        tf_names = load_tf_names(i + "/transcription_factors.tsv")
        network = grnboost2(expression_data=ex_matrix, gene_names=gene_name, tf_names=tf_names).iloc[:, :2]
        print(network)
        network.to_csv('/home/fengke/pycode/my_exp/paper/exp/output/grnboost2/' + i.split("/")[-1] + ".txt", sep='\t',
                       index=False, header=False)


def run_narromi():
    """
    Run narromi to infer GRN.
    """
    original_dir = utils.get_dir_list()[:3]
    for i in original_dir:
        print(i.split("/")[-1])
        ex_matrix = pd.read_csv(i + "/expression_data.tsv", sep='\t', header=0)
        tmp_exp = "tmp/" + i.split("/")[-1] + "_tmp_expression_data.tsv"
        ex_matrix.to_csv(tmp_exp, sep="\t", header=False, index=False)
        tmp_genes = "tmp/" + i.split("/")[-1] + "_tmp_genes.txt"
        pd.Series(ex_matrix.columns).to_csv(tmp_genes, sep="\n", header=False, index=False)
        os.system(
            "/home/fengke/seidr/bin/narromi -i " + tmp_exp + " -g " + tmp_genes + " -t " + i + "/transcription_factors.tsv" + " -o " + '/home/fengke/pycode/my_exp/paper/exp/output/narromi/' +
            i.split("/")[-1] + ".txt")
        res = pd.read_csv('/home/fengke/pycode/my_exp/paper/exp/output/narromi/' +
                          i.split("/")[-1] + ".txt", sep="\t", header=None)
        res[2] = np.abs(res[2])
        res.sort_values(by=2, ascending=False, inplace=True)
        print(res)
        res.iloc[:, :2].to_csv('/home/fengke/pycode/my_exp/paper/exp/output/narromi/' +
                               i.split("/")[-1] + ".txt", sep="\t", header=False, index=False)


def convert_PIDC():
    """
    Convert PIDC result.
    """
    base_input_dir = "/home/fengke/pycode/my_exp/paper/exp/input/"
    sc_names = ["hESC", "mDC"]
    gs_names = ["Cell_type_specific", "Non_specific", "STRING"]
    num_names = ["500", "1000"]
    for sc in sc_names:
        for gs in gs_names:
            for num in num_names:
                ex_matrix = pd.read_csv(base_input_dir + sc + "/" + gs + "/" + num + "/expression_data.tsv", sep='\t',
                                        header=0)
                tmp_output = "/home/fengke/pycode/my_exp/paper/exp/tmp/" + sc + "_" + gs + "_" + num + ".txt"
                ex_matrix.T.to_csv(tmp_output, sep="\t", header=True, index=True)
    # original_dir = utils.get_dir_list()[:3]
    # for i in original_dir:
    #     net_name = i.split("/")[-1]
    #     print(net_name)
    #     ex_matrix = pd.read_csv(i + "/expression_data.tsv", sep='\t', header=0)
    #     print(ex_matrix.T)
    #     ex_matrix.T.to_csv("tmp/" + net_name + "_PIDC.txt", sep="\t", header=True, index=True)


def delete_non_tfs():
    """
    Delete non-TF items in PIDC results
    """
    dir = "/home/fengke/pycode/my_exp/paper/exp/output/DREAM5_alg_output/PIDC"
    net_name_list = ["e_coli"]
    for net_name in net_name_list:
        tfs = list(pd.read_csv("/home/fengke/pycode/my_exp/paper/exp/input/" + net_name + "/transcription_factors.tsv",
                               sep="\t", header=None)[0])
        print(tfs)
        data = pd.read_csv(dir + "/" + net_name + ".txt", sep="\t", header=None)
        data = data[data[0].isin(tfs)]
        data.to_csv(dir + "/" + net_name + ".txt", sep="\t", header=False, index=False)


def post_porcess_res_of_ppcor():
    """
    Process the result of the ppcor method.
    """
    dir = "/home/fengke/pycode/my_exp/paper/exp/output/ppcor"
    files = os.listdir(dir)
    for file in files:
        print(file)
        if "in_silico" in file:
            net_name = "in_silico"
        elif "e_coli" in file:
            net_name = "e_coli"
        elif "s_cere" in file:
            net_name = "s_cere"
        tf_num = pd.read_csv("/home/fengke/pycode/my_exp/paper/exp/input/" + net_name + "/transcription_factors.tsv",
                             sep="\t", header=None).shape[0]
        data = pd.read_csv(dir + "/" + file, sep="\t", header=None)
        genes = pd.Index(["G" + str(i + 1) for i in range(data.shape[0])])
        print(genes)
        res = utils.generate_res(data.values, tf_num, genes)
        res = res.iloc[:, :2]
        res.to_csv(dir + "/" + net_name + ".txt", sep="\t", header=False, index=False)
        print(res)


def delete_conf():
    """
    Remove the last column in the ranked list
    """
    # net_name_list = ["e_coli.txt"]
    dir = "/home/fengke/pycode/my_exp/paper/exp/output/sc_alg_output/PIDC/"
    for file in os.listdir(dir):
        data = pd.read_csv(dir + file, sep="\t", header=None).iloc[:, :2]
        print(data)
        data.to_csv(dir + file, sep="\t", header=False, index=False)


def run_eval():
    """
    Evaluation of various results.
    """
    alg_list = ["plsnet"]
    net_name_list = ["in_silico", "e_coli", "s_cere"]
    dir = "/home/fengke/pycode/my_exp/paper/exp/output/DREAM5_alg_output/"
    for alg in alg_list:
        for net_name in net_name_list:
            pred = pd.read_csv(dir + alg + "/" + net_name + ".txt", sep="\t", header=None)
            gs = pd.read_csv("/home/fengke/pycode/my_exp/paper/exp/input/" + net_name + "/" + "gold_standard.tsv",
                             sep="\t", header=None)
            pdf_aupr = scio.loadmat("/home/fengke/pycode/my_exp/paper/exp/input/" + net_name + "/" + "AUPR.mat")
            pdf_auroc = scio.loadmat("/home/fengke/pycode/my_exp/paper/exp/input/" + net_name + "/" + "AUROC.mat")
            auroc, aupr, conf_score, epr = eval.run(prediction=pred, gold_standard=gs, pdf_aupr=pdf_aupr,
                                                    pdf_auroc=pdf_auroc)
            print(alg, "   ", net_name)
            print(auroc, aupr, conf_score, epr)


def run_sc_alg():
    """
    Run PLSNET to infer GRN.
    """
    base_input_dir = "/home/fengke/pycode/my_exp/paper/exp/input/"
    # sc_names = ["hESC", "mDC"]
    # gs_names = ["Cell_type_specific", "Non_specific", "STRING"]
    # num_names = ["500", "1000"]
    net_name = ["in_silico", "e_coli", "s_cere"]
    alg_name = "plsnet"
    for net in net_name:
        data = pd.read_csv(base_input_dir + net + "/expression_data.tsv", sep="\t",
                           header=0)
        tmp_input_file = "/home/fengke/pycode/my_exp/paper/exp/tmp/" + net + ".tsv"
        data.to_csv(tmp_input_file, sep="\t", header=False, index=False)

        genes = pd.read_csv(base_input_dir + net + "/gene_ids.tsv", sep="\t",
                            header=0).iloc[:, 0]
        tmp_genes_file = "/home/fengke/pycode/my_exp/paper/exp/tmp/" + net + "_genes.txt"
        genes.to_csv(tmp_genes_file, sep="\t", header=False, index=False)

        tf_file = base_input_dir + net + "/transcription_factors.tsv"
        output_file = "/home/fengke/pycode/my_exp/paper/exp/output/DREAM5_alg_output/" + alg_name + "/" + net + ".txt"
        command = "/home/fengke/seidr/bin/plsnet -i " + tmp_input_file + " -g " + tmp_genes_file + " -t " + tf_file + " -o " + output_file
        os.system(command)
        res = pd.read_csv(output_file, sep="\t", header=None)
        print(res)
        res[2] = res[2].abs()
        res.sort_values(by=2, ascending=False, inplace=True)
        res.iloc[:, :2].to_csv(output_file, sep="\t", header=False, index=False)
        print("------------------------------")
        # for sc in sc_names:
        #     for gs in gs_names:
        #         for num in num_names:
        # ex_matrix = pd.read_csv(base_input_dir + sc + "/" + gs + "/" + num + "/expression_data.tsv", sep='\t',
        #                         header=0)
        # gene_name = list(ex_matrix.columns)
        # # tf_names is read using a utility function included in Arboreto
        # tf_names = load_tf_names(base_input_dir + sc + "/" + gs + "/" + num + "/transcription_factors.tsv")
        # network = grnboost2(expression_data=ex_matrix, gene_names=gene_name, tf_names=tf_names)
        # print(network)
        # network = network.iloc[:, :2]
        # network.to_csv(
        #     "/home/fengke/pycode/my_exp/paper/exp/output/sc_alg_output/" + alg_name + "/" + sc + "_" + gs + "_" + num + ".txt",
        #     sep='\t',
        #     index=False, header=False)
        # ----------------------------------------
        # data = pd.read_csv(base_input_dir + sc + "/" + gs + "/" + num + "/expression_data.tsv", sep="\t",
        #                    header=0)
        # tmp_input_file = "/home/fengke/pycode/my_exp/paper/exp/tmp/" + sc + "_" + gs + "_" + num + ".tsv"
        # data.to_csv(tmp_input_file, sep="\t", header=False, index=False)
        #
        # genes = pd.read_csv(base_input_dir + sc + "/" + gs + "/" + num + "/gene_ids.tsv", sep="\t",
        #                     header=0).iloc[:, 0]
        # tmp_genes_file = "/home/fengke/pycode/my_exp/paper/exp/tmp/" + sc + "_" + gs + "_" + num + "_genes.txt"
        # genes.to_csv(tmp_genes_file, sep="\t", header=False, index=False)
        #
        # tf_file = base_input_dir + sc + "/" + gs + "/" + num + "/transcription_factors.tsv"
        # output_file = "/home/fengke/pycode/my_exp/paper/exp/output/sc_alg_output/" + alg_name + "/" + sc + "_" + gs + "_" + num + ".txt"
        # command = "/home/fengke/seidr/bin/tigress -i " + tmp_input_file + " -g " + tmp_genes_file + " -t " + tf_file + " -o " + output_file
        # os.system(command)
        # res = pd.read_csv(output_file, sep="\t", header=None)
        # print(res)
        # res[2] = res[2].abs()
        # res.sort_values(by=2, ascending=False, inplace=True)
        # res.iloc[:, :2].to_csv(output_file, sep="\t", header=False, index=False)
        # print("------------------------------")


def sub_PIDC(cmd):
    os.system(cmd)


def run_PIDC():
    """
    Run PIDC to infer GRN.
    """
    base_input_dir = "/home/fengke/pycode/my_exp/paper/exp/input/"
    jl_script = "/home/fengke/pycode/my_exp/paper/exp/PIDC.jl"
    sc_names = ["hESC", "mDC"]
    gs_names = ["Cell_type_specific", "Non_specific", "STRING"]
    num_names = ["500", "1000"]
    p_list = []
    for sc in sc_names:
        for gs in gs_names:
            for num in num_names:
                command = "julia " + jl_script + " " + sc + "_" + gs + "_" + num
                # print(command)
                p = Process(target=sub_PIDC, args=(command,))
                p_list.append(p)
                p.start()
    for i in p_list:
        i.join()


def run_sc_eval():
    """
    Evaluation of single cell resluts.
    """
    base_input_dir = "/home/fengke/pycode/my_exp/paper/exp/input/"
    base_out_dir = "/home/fengke/pycode/my_exp/paper/exp/output/sc_alg_output/"
    alg_list = os.listdir(base_out_dir)
    sc_names = ["hESC", "mDC"]
    gs_names = ["Cell_type_specific", "Non_specific", "STRING"]
    num_names = ["500", "1000"]
    res_df = pd.DataFrame(columns=["alg", "sc", "gs", "num", "auroc", "aupr", "conf_score", "epr"])
    for alg in tqdm(alg_list):
        print(alg)
        for sc in sc_names:
            for gs_name in gs_names:
                for num in num_names:
                    input_dir = base_input_dir + sc + "/" + gs_name + "/" + num + "/"
                    pred_file = base_out_dir + alg + "/" + sc + "_" + gs_name + "_" + num + ".txt"
                    pred = pd.read_csv(pred_file, sep="\t", header=None)
                    print(input_dir)
                    gs = pd.read_csv(input_dir + "gold_standard.tsv", sep="\t", header=None)
                    pdf_aupr = scio.loadmat(input_dir + "AUPR.mat")
                    pdf_auroc = scio.loadmat(input_dir + "AUROC.mat")
                    auroc, aupr, conf_score, epr = eval.run(prediction=pred, gold_standard=gs, pdf_aupr=pdf_aupr,
                                                            pdf_auroc=pdf_auroc)
                    res_dic = {"alg": alg, "sc": sc, "gs": gs_name, "num": num, "auroc": auroc, "aupr": aupr,
                               "conf_score": conf_score, "epr": epr}
                    res_df = res_df.append(res_dic, ignore_index=True)
                    print("-----------------------------------")
    print(res_df)
    res_df.to_csv("sc_alg_eval_res.csv", sep=",", header=True, index=True)


def our_res():
    base_out_dir = "/home/fengke/pycode/my_exp/paper/exp/output/"
    sc_names = ["hESC", "mDC"]
    gs_names = ["Cell_type_specific", "Non_specific", "STRING"]
    num_names = ["500", "1000"]
    res_df = pd.DataFrame(columns=["sc", "gs", "num", "auroc", "aupr", "conf_score", "epr"])
    for sc in sc_names:
        for gs_name in gs_names:
            for num in num_names:
                auroc_list = []
                aupr_list = []
                score_list = []
                epr_list = []
                dir = base_out_dir + sc + "/" + gs_name + "/" + num + "/"
                res_list = [res for res in os.listdir(dir) if res.startswith("res")]
                for res in res_list:
                    with open(dir + "/" + res, 'r+') as f:
                        auroc_list.append(float(f.readline()))
                        aupr_list.append(float(f.readline()))
                        score_list.append(float(f.readline()))
                        epr_list.append(float(f.readline()))
                res_dic = {"sc": sc, "gs": gs_name, "num": num, "auroc": np.mean(auroc_list),
                           "aupr": np.mean(aupr_list), "conf_score": np.mean(score_list), "epr": np.mean(epr_list)}
                res_df = res_df.append(res_dic, ignore_index=True)
    print(res_df)
    res_df.to_csv("our_eval_res.csv", sep=",", header=True, index=True)


def eval_scribe():
    """
    Evaluation of Sribe.
    """
    sc_names = ["hESC", "mDC"]
    gs_names = ["Cell_type_specific", "Non_specific", "STRING"]
    num_names = ["500", "1000"]
    output_dir = "~/pycode/my_exp/Beeline/Beeline-master/outputs/"
    res_df = pd.DataFrame(columns=["sc", "gs", "num", "auroc", "aupr", "conf_score", "epr"])
    for sc in sc_names:
        for gs_name in gs_names:
            for num in num_names:
                gene_mapping = pd.read_csv(
                    "./input/" + sc + "/" + gs_name + "/" + num + "/gene_ids.tsv", sep="\t",
                    header=0)
                gs = pd.read_csv(
                    "./input/" + sc + "/" + gs_name + "/" + num + "/gold_standard.tsv", sep="\t",
                    header=None)
                pdf_auroc = scio.loadmat("./input/" + sc + "/" + gs_name + "/" + num + "/AUROC.mat")
                pdf_aupr = scio.loadmat("./input/" + sc + "/" + gs_name + "/" + num + "/AUPR.mat")
                mapping_dict = gene_mapping.set_index(["Name"])["#ID"].to_dict()
                pred_file = output_dir + sc + "/" + gs_name + "/" + num + "/SCRIBE/" + "rankedEdges.csv"
                pred = pd.read_csv(pred_file, sep="\t", header=0).iloc[:, :2]
                pred = pred[
                    (pred["Gene1"].isin(list(gene_mapping["Name"]))) & (pred["Gene2"].isin(list(gene_mapping["Name"])))]
                pred["Gene1"] = pred["Gene1"].map(mapping_dict)
                pred["Gene2"] = pred["Gene2"].map(mapping_dict)
                pred.columns = [0, 1]
                # print(pred)
                auroc, aupr, conf_score, epr = eval.run(prediction=pred, gold_standard=gs, pdf_auroc=pdf_auroc,
                                                        pdf_aupr=pdf_aupr)
                res_dic = {"sc": sc, "gs": gs_name, "num": num, "auroc": auroc, "aupr": aupr,
                           "conf_score": conf_score, "epr": epr}
                res_df = res_df.append(res_dic, ignore_index=True)
                print(auroc, aupr, conf_score, epr)
                print("------------------------------")
    print(res_df)
    res_df.to_csv("scribe_res.csv", sep=",", header=True, index=True)


def eval_ANM():
    """
    Evaluation of isolated ANM.
    """
    base_dir = './input/'
    dream5_data = ['e_coli', 'in_silico', 's_cere']
    sc_name = ['hESC', 'mDC']
    gs_type = ['Cell_type_specific', 'Non_specific', 'STRING']
    num_list = ["500", "1000"]
    sc_data = []
    for i in sc_name:
        for j in gs_type:
            for k in num_list:
                sc_data.append(i + "/" + j + "/" + k)
    print(dream5_data)
    print(sc_data)
    for m in dream5_data:
        tf_list = []
        tg_list = []
        conf_list = []
        anm = ANM()
        tfs = list(pd.read_csv(base_dir + m + "/" + "transcription_factors.tsv", sep="\t", header=None)[0])
        gs = pd.read_csv(base_dir + m + "/" + "gold_standard.tsv", sep="\t", header=0)
        pdf_auroc = scio.loadmat(base_dir + m + "/" + "AUROC.mat")
        pdf_aupr = scio.loadmat(base_dir + m + "/" + "AUPR.mat")
        exp = pd.read_csv(base_dir + m + "/" + "expression_data.tsv", sep="\t", header=0)
        for tf in tqdm(tfs):
            for tg in exp.columns:
                if tf == tg:
                    continue
                tf_list.append(tf)
                tg_list.append(tg)
                score = anm.predict_proba((exp[tf].values, exp[tg].values))
                conf_list.append(score)
        pred = pd.DataFrame({'tf': tf_list, 'tg': tg_list, 'conf': conf_list})
        pred = pred.sort_values(by='conf', ascending=False)
        pred = pred.iloc[:, 0:2]
        auroc, aupr, conf_score, epr = eval.run(pred, gs, pdf_auroc=pdf_auroc, pdf_aupr=pdf_aupr)
        print(m)
        print(auroc, aupr, conf_score, epr)
        break


def draw_dream5_res(sheet_name, x_lim):
    data = pd.read_excel("E:\\sync\\OneDrive - for personal\\wzwzwzwz\\res_for_chart\\dream5_result.xlsx",
                         sheet_name=sheet_name, header=0, index_col=0).T
    data = data.iloc[::-1]
    plt.rcParams.update({'figure.autolayout': True})
    plt.style.use("fivethirtyeight")
    color_list = ["#343830", "#03719C", "#0F9B8E"]
    fig, ax = plt.subplots(figsize=(11, 30))
    fig.set_facecolor("white")
    ax.set_facecolor("white")
    ax.spines['right'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')
    plt.grid(False)
    yticks = np.array(range(data.shape[0])) * 2.2
    height = 0.6
    interval = 0.05

    ax.barh(yticks, data["In Silico"], height=height, label="In Silico", color=color_list[0])
    for idx in range(data.shape[0]):
        ax.text(data["In Silico"][idx] + 0.002, yticks[idx] - 0.2, data["In Silico"][idx], fontsize=10)
    ax.barh(yticks + height + interval, data["E.coli"], height=height, label="E.coli", color=color_list[1])
    for idx in range(data.shape[0]):
        tmp = yticks + height + interval
        ax.text(data["E.coli"][idx] + 0.002, tmp[idx] - 0.2, data["E.coli"][idx], fontsize=10)
    ax.barh(yticks + 2 * (height + interval), data["S.cere"], height=height, label="S.cere", color=color_list[2])
    for idx in range(data.shape[0]):
        tmp = yticks + 2 * (height + interval)
        ax.text(data["S.cere"][idx] + 0.002, tmp[idx] - 0.2, data["S.cere"][idx], fontsize=10)
    plt.yticks(yticks + height + interval, data.index)
    line1 = data.iloc[-1, 0]
    line2 = data.iloc[-1, 1]
    line3 = data.iloc[-1, 2]
    ax.set(xlim=x_lim, xlabel=sheet_name, ylabel='Methods')
    # Add a vertical line, here we set the style in the function call
    ax.axvline(line1, ymin=0.02, ymax=0.98, ls=':', linewidth=1.5, color=color_list[0])
    ax.axvline(line2, ymin=0.02, ymax=0.98, ls=':', linewidth=1.5, color=color_list[1])
    ax.axvline(line3, ymin=0.02, ymax=0.98, ls=':', linewidth=1.5, color=color_list[2])
    ax.legend(bbox_to_anchor=(0.82, 0.14), frameon=False)
    plt.show()


def draw_dream5_res2(sheet_name, y_lim):
    data = pd.read_excel("E:\\sync\\OneDrive - for personal\\wzwzwzwz\\res_for_chart\\dream5_result.xlsx",
                         sheet_name=sheet_name, header=0, index_col=0).T
    print(data)
    # data = data.iloc[::-1]
    plt.rcParams.update({'figure.autolayout': True})
    plt.style.use("fivethirtyeight")
    color_list = ["#ea5f50", "#eddba4", "#52999f"]
    fig, ax = plt.subplots(figsize=(25, 8))
    fig.set_facecolor("white")
    ax.set_facecolor("white")
    ax.spines['right'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    plt.grid(False)
    xticks = np.array(range(data.shape[0])) * 2.4
    width = 0.62
    interval = 0.05

    ax.bar(xticks, data["In Silico"], width=width, label="In Silico", color=color_list[0])
    # for idx in range(data.shape[0]):
    #     ax.text(data["In Silico"][idx] + 0.002, xticks[idx] - 0.2, data["In Silico"][idx], fontsize=10)
    ax.bar(xticks + width + interval, data["E.coli"], width=width, label="E.coli", color=color_list[1])
    # for idx in range(data.shape[0]):
    #     tmp = xticks + width + interval
    #     ax.text(data["E.coli"][idx] + 0.002, tmp[idx] - 0.2, data["E.coli"][idx], fontsize=10)
    ax.bar(xticks + 2 * (width + interval), data["S.cere"], width=width, label="S.cere", color=color_list[2])
    # for idx in range(data.shape[0]):
    #     tmp = xticks + 2 * (width + interval)
    #     ax.text(data["S.cere"][idx] + 0.002, tmp[idx] - 0.2, data["S.cere"][idx], fontsize=10)
    plt.xticks(xticks, data.index, rotation=290, ha='left')
    line1 = data.iloc[0, 0]
    line2 = data.iloc[0, 1]
    line3 = data.iloc[0, 2]
    ax.set(ylim=y_lim)
    # Add a vertical line, here we set the style in the function call
    ax.axhline(line1, xmin=0.048, xmax=0.96, ls='-.', linewidth=2, color=color_list[0])
    ax.axhline(line2, xmin=0.06, xmax=0.96, ls='-.', linewidth=2, color=color_list[1])
    ax.axhline(line3, xmin=0.072, xmax=0.96, ls='-.', linewidth=2, color=color_list[2])
    ax.text(xticks[0] - 0.2, data["In Silico"][0] + 0.005, data["In Silico"][0], fontsize=12)
    ax.text(xticks[0] + width - 0.2, data["E.coli"][0] + 0.005, data["E.coli"][0], fontsize=12)
    ax.text(xticks[0] + 2 * width - 0.2, data["S.cere"][0] + 0.005, data["S.cere"][0], fontsize=12)
    # ax.legend(bbox_to_anchor=(0.82, 0.14), frameon=False)
    ax.legend(frameon=False)
    plt.title(sheet_name + " (DREAM5)", fontsize='large', fontweight='bold')
    plt.savefig('demo.svg', facecolor='white', dpi=120)
    # plt.show()


def draw_heatmap_for_dream5():
    score = pd.read_excel("E:\\sync\\OneDrive - for personal\\wzwzwzwz\\res_for_chart\\TF_TF_dream5_result.xlsx",
                          sheet_name="Confidence Score", header=0, index_col=0).T
    epr = pd.read_excel("E:\\sync\\OneDrive - for personal\\wzwzwzwz\\res_for_chart\\TF_TF_dream5_result.xlsx",
                        sheet_name="EPR", header=0, index_col=0).T
    sns.set(rc={'figure.figsize': (12, 8)})
    plt.rcParams.update({'figure.autolayout': True})
    plt.subplot(1, 2, 1)
    sns.heatmap(score, annot=True, fmt=".3f", robust="True")
    plt.title("Confidence Score")
    plt.subplot(1, 2, 2)
    ax = sns.heatmap(epr, annot=True, fmt=".3f", robust="True", cmap="YlGnBu_r")
    ax.axes.yaxis.set_visible(False)
    plt.title("EPR")
    plt.subplots_adjust(wspace=0.25)
    plt.show()


def draw_sc_res_auroc(datasets_name, sheet_name, x_lim):
    if datasets_name == "hESC":

        data = pd.read_excel("E:\\sync\\OneDrive - for personal\\wzwzwzwz\\res_for_chart\\sc_result.xlsx",
                             sheet_name=sheet_name, header=0, index_col=[0, 1, 2]).iloc[0:6, :].T
    elif datasets_name == "mDC":
        data = pd.read_excel("E:\\sync\\OneDrive - for personal\\wzwzwzwz\\res_for_chart\\sc_result.xlsx",
                             sheet_name=sheet_name, header=0, index_col=[0, 1, 2]).iloc[6:12, :].T
    data = data.iloc[::-1]
    print(data)
    data500 = data.iloc[:, 0:3]
    data1000 = data.iloc[:, 3:6]
    plt.rcParams.update({'figure.autolayout': True})
    plt.style.use("fivethirtyeight")
    color_list = ["#9c9797", "#cd9692", "#deb69c"]
    fig, ax = plt.subplots(figsize=(12, 35), facecolor='white')
    ax.set_facecolor("white")
    ax.spines['right'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')
    plt.grid(False)
    yticks = np.array(range(data.shape[0])) * 2.5
    height = 0.6
    interval = 0.05
    labels = ["Cell-type-specific", "Non-specific", "STRING"]
    # plt.legend(frameon=False)
    for i in range(3):
        ax.barh(yticks + i * (height + interval), -np.array(data500.iloc[:, i] - x_lim[0]), height=height,
                label=labels[i],
                color=color_list[i])
        ax.barh(yticks + i * (height + interval), data1000.iloc[:, i] - x_lim[0], height=height, label=labels[i],
                color=color_list[i])
        # for idx in range(data.shape[0]):
        #     tmp = yticks + i * (height + interval)
        #     ax[0].text(data.iloc[:, i][idx] + 0.002, tmp[idx] - 0.2, round(data.iloc[:, i][idx], 3), fontsize=10)
    ax.grid(False)
    ax.set_yticks(yticks + height + interval)
    ax.set_yticklabels(data.index)
    # print(ax.get_xticklabels())
    ax.set_xticklabels(np.around(np.abs(ax.get_xticks()) + x_lim[0], decimals=2))
    # print(ax.get_xticklabels())
    line1 = data1000.iloc[-1, 0] - x_lim[0]
    line2 = data1000.iloc[-1, 1] - x_lim[0]
    line3 = data1000.iloc[-1, 2] - x_lim[0]
    _line1 = -1 * (data500.iloc[-1, 0] - x_lim[0])
    _line2 = -1 * (data500.iloc[-1, 1] - x_lim[0])
    _line3 = -1 * (data500.iloc[-1, 2] - x_lim[0])

    # ax.set_xlim([(x_lim[0] - x_lim[1]), (x_lim[1] - x_lim[0])])
    # ax.set_xlabel(sheet_name)
    # ax.set_ylabel("Methods")
    ax.set_title(sheet_name + " (" + datasets_name + ")", fontsize='large', fontweight='bold')
    ax.text(-1 * (x_lim[1] - x_lim[0]) / 2 + 0.01, yticks[-1] + 2, "TFs + 500 genes", fontsize=15)
    ax.text((x_lim[1] - x_lim[0]) / 2 - 0.09, yticks[-1] + 2, "TFs + 1,000 genes", fontsize=15)
    # Add a vertical line, here we set the style in the function call
    ax.axvline(0, ymin=0.02, ymax=0.98, ls='solid', linewidth=3, color='white')
    ax.axvline(line1, ymin=0.04, ymax=0.915, ls='-.', linewidth=2, color=color_list[0])
    ax.axvline(line2, ymin=0.04, ymax=0.935, ls='-.', linewidth=2, color=color_list[1])
    ax.axvline(line3, ymin=0.04, ymax=0.955, ls='-.', linewidth=2, color=color_list[2])
    ax.axvline(_line1, ymin=0.04, ymax=0.915, ls='-.', linewidth=2, color=color_list[0])
    ax.axvline(_line2, ymin=0.04, ymax=0.935, ls='-.', linewidth=2, color=color_list[1])
    ax.axvline(_line3, ymin=0.04, ymax=0.955, ls='-.', linewidth=2, color=color_list[2])
    ########################################
    ax.text(_line1 + 0.004, yticks[-1] - 0.2, round(_line1 * -1 + x_lim[0], 3), fontsize=10)
    ax.text(_line2 + 0.004, yticks[-1] - 0.2 + height + interval, round(_line2 * -1 + x_lim[0], 3), fontsize=10)
    ax.text(_line3 + 0.004, yticks[-1] - 0.22 + 2 * (height + interval), round(_line3 * -1 + x_lim[0], 3), fontsize=10)
    ax.text(line1 - 0.016, yticks[-1] - 0.2, round(line1 + x_lim[0], 3), fontsize=10)
    ax.text(line2 - 0.014, yticks[-1] - 0.2 + height + interval, round(line2 + x_lim[0], 3), fontsize=10)
    ax.text(line3 - 0.016, yticks[-1] - 0.22 + 2 * (height + interval), round(line3 + x_lim[0], 3), fontsize=10)
    # ---------------------------------
    # for i in range(3):
    #     ax[1].barh(yticks + i * (height + interval), data.iloc[:, i + 3], height=height, label=labels[i],
    #                color=color_list[i])
    #     for idx in range(data.shape[0]):
    #         tmp = yticks + i * (height + interval)
    #         ax[1].text(data.iloc[:, i + 3][idx] + 0.002, tmp[idx] - 0.2, round(data.iloc[:, i + 3][idx], 3),
    #                    fontsize=10)
    # ax[1].grid(False)
    # line1 = data.iloc[-1, 3]
    # line2 = data.iloc[-1, 4]
    # line3 = data.iloc[-1, 5]
    # ax[1].axes.yaxis.set_visible(False)
    # ax[1].set_xlim(x_lim)
    # ax[1].set_xlabel(sheet_name)
    # ax[1].set_title(datasets_name + "(1000)")
    # # Add a vertical line, here we set the style in the function call
    # ax[1].axvline(line1, ymin=0.02, ymax=0.98, ls=':', linewidth=1.5, color=color_list[0])
    # ax[1].axvline(line2, ymin=0.02, ymax=0.98, ls=':', linewidth=1.5, color=color_list[1])
    # ax[1].axvline(line3, ymin=0.02, ymax=0.98, ls=':', linewidth=1.5, color=color_list[2])
    # ax[1].legend(bbox_to_anchor=(0.9, 0.14), frameon=False)
    # ax.legend(frameon=False)
    plt.subplots_adjust(wspace=0.5)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[0:6:2], labels[0:6:2], frameon=False, bbox_to_anchor=(1, 0.05), loc=3, borderaxespad=0)
    # plt.savefig('demo.svg', facecolor='white')
    plt.show()


def transf(x):
    return (x + 1)


def draw_sc_res_aupr(datasets_name, sheet_name, x_lim):
    if datasets_name == "hESC":

        data = pd.read_excel("E:\\sync\\OneDrive - for personal\\wzwzwzwz\\res_for_chart\\sc_result.xlsx",
                             sheet_name=sheet_name, header=0, index_col=[0, 1, 2]).iloc[0:6, :].T
    elif datasets_name == "mDC":
        data = pd.read_excel("E:\\sync\\OneDrive - for personal\\wzwzwzwz\\res_for_chart\\sc_result.xlsx",
                             sheet_name=sheet_name, header=0, index_col=[0, 1, 2]).iloc[6:12, :].T
    data = data.iloc[::-1]
    data500 = data.iloc[:, 0:3]
    data1000 = data.iloc[:, 3:6]
    plt.rcParams.update({'figure.autolayout': True})
    plt.style.use("fivethirtyeight")
    color_list = ["#9c9797", "#cd9692", "#deb69c"]
    fig, ax = plt.subplots(figsize=(12, 35), facecolor='white')
    ax.set_facecolor("white")
    ax.spines['right'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')
    plt.grid(False)
    yticks = np.array(range(data.shape[0])) * 2.5
    height = 0.6
    interval = 0.05
    labels = ["Cell-type-specific", "Non-specific", "STRING"]
    # plt.legend(frameon=False)
    for i in range(3):
        ax.barh(yticks + i * (height + interval), -np.array(data500.iloc[:, i] - x_lim[0]), height=height,
                label=labels[i],
                color=color_list[i])
        ax.barh(yticks + i * (height + interval), data1000.iloc[:, i] - x_lim[0], height=height, label=labels[i],
                color=color_list[i])
        # for idx in range(data.shape[0]):
        #     tmp = yticks + i * (height + interval)
        #     ax[0].text(data.iloc[:, i][idx] + 0.002, tmp[idx] - 0.2, round(data.iloc[:, i][idx], 3), fontsize=10)
    ax.grid(False)
    ax.set_yticks(yticks + height + interval)
    ax.set_yticklabels(data.index)
    # ax.set_xticks([-0.20, -0.15, -0.10, - 0.05, 0.0, 0.05, 0.10, 0.15, 0.20])
    # ax.set_xticklabels(["0.20", "0.15", "0.10", "0.05", "0", "0.05", "0.10", "0.15", "0.20"])
    ax.set_xticks([-0.100, -0.075, -0.050, -0.025, 0, 0.025, 0.050, 0.075, 0.100])
    ax.set_xticklabels(["0.100", "0.075", "0.050", "0.025", "0", "0.025", "0.050", "0.075", "0.100"])
    # print(ax.get_xticks())
    line1 = data1000.iloc[-1, 0] - x_lim[0]
    line2 = data1000.iloc[-1, 1] - x_lim[0]
    line3 = data1000.iloc[-1, 2] - x_lim[0]
    _line1 = -1 * (data500.iloc[-1, 0] - x_lim[0])
    _line2 = -1 * (data500.iloc[-1, 1] - x_lim[0])
    _line3 = -1 * (data500.iloc[-1, 2] - x_lim[0])

    # ax.set_xlim([(x_lim[0] - x_lim[1]), (x_lim[1] - x_lim[0])])
    # ax.set_xlabel(sheet_name)
    # ax.set_ylabel("Methods")
    ax.set_title(sheet_name + " (" + datasets_name + ")", fontsize='large', fontweight='bold')
    ax.text(-0.075, yticks[-1] + 2, "TFs + 500 genes", fontsize=15)
    ax.text(0.015, yticks[-1] + 2, "TFs + 1,000 genes", fontsize=15)
    # Add a vertical line, here we set the style in the function call
    ax.axvline(0, ymin=0.02, ymax=0.98, ls='solid', linewidth=3, color='white')
    ax.axvline(line1, ymin=0.04, ymax=0.915, ls='-.', linewidth=2, color=color_list[0])
    ax.axvline(line2, ymin=0.04, ymax=0.935, ls='-.', linewidth=2, color=color_list[1])
    ax.axvline(line3, ymin=0.04, ymax=0.955, ls='-.', linewidth=2, color=color_list[2])
    ax.axvline(_line1, ymin=0.04, ymax=0.915, ls='-.', linewidth=2, color=color_list[0])
    ax.axvline(_line2, ymin=0.04, ymax=0.935, ls='-.', linewidth=2, color=color_list[1])
    ax.axvline(_line3, ymin=0.04, ymax=0.955, ls='-.', linewidth=2, color=color_list[2])
    ########################################
    ax.text(_line1 + 0.001, yticks[-1] - 0.2, round(_line1 * -1 + x_lim[0], 3), fontsize=10)
    ax.text(_line2 + 0.001, yticks[-1] - 0.2 + height + interval, round(_line2 * -1 + x_lim[0], 3), fontsize=10)
    ax.text(_line3 + 0.001, yticks[-1] - 0.22 + 2 * (height + interval), round(_line3 * -1 + x_lim[0], 3),
            fontsize=10)
    ax.text(line1 - 0.014, yticks[-1] - 0.2, round(line1 + x_lim[0], 3), fontsize=10)
    ax.text(line2 - 0.014, yticks[-1] - 0.2 + height + interval, round(line2 + x_lim[0], 3), fontsize=10)
    ax.text(line3 - 0.014, yticks[-1] - 0.22 + 2 * (height + interval), round(line3 + x_lim[0], 3), fontsize=10)
    # ---------------------------------
    # for i in range(3):
    #     ax[1].barh(yticks + i * (height + interval), data.iloc[:, i + 3], height=height, label=labels[i],
    #                color=color_list[i])
    #     for idx in range(data.shape[0]):
    #         tmp = yticks + i * (height + interval)
    #         ax[1].text(data.iloc[:, i + 3][idx] + 0.002, tmp[idx] - 0.2, round(data.iloc[:, i + 3][idx], 3),
    #                    fontsize=10)
    # ax[1].grid(False)
    # line1 = data.iloc[-1, 3]
    # line2 = data.iloc[-1, 4]
    # line3 = data.iloc[-1, 5]
    # ax[1].axes.yaxis.set_visible(False)
    # ax[1].set_xlim(x_lim)
    # ax[1].set_xlabel(sheet_name)
    # ax[1].set_title(datasets_name + "(1000)")
    # # Add a vertical line, here we set the style in the function call
    # ax[1].axvline(line1, ymin=0.02, ymax=0.98, ls=':', linewidth=1.5, color=color_list[0])
    # ax[1].axvline(line2, ymin=0.02, ymax=0.98, ls=':', linewidth=1.5, color=color_list[1])
    # ax[1].axvline(line3, ymin=0.02, ymax=0.98, ls=':', linewidth=1.5, color=color_list[2])
    # ax[1].legend(bbox_to_anchor=(0.9, 0.14), frameon=False)
    # ax.legend(frameon=False)
    plt.subplots_adjust(wspace=0.5)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[0:6:2], labels[0:6:2], frameon=False, bbox_to_anchor=(1, 0.05), loc=3, borderaxespad=0)
    # plt.savefig('demo.svg', facecolor='white')
    plt.show()


def draw_heatmap_for_sc(data_name):
    score = pd.read_excel("E:\\sync\\OneDrive - for personal\\wzwzwzwz\\res_for_chart\\TF_TF_sc_result.xlsx",
                          sheet_name="Confidence Score", header=0, index_col=[0, 1, 2]).T
    epr = pd.read_excel("E:\\sync\\OneDrive - for personal\\wzwzwzwz\\res_for_chart\\TF_TF_sc_result.xlsx",
                        sheet_name="EPR", header=0, index_col=[0, 1, 2]).T
    if data_name == "mDC":
        score = score.iloc[:, 6:12]
        epr = epr.iloc[:, 6:12]
    else:
        score = score.iloc[:, 0:6]
        epr = epr.iloc[:, 0:6]
    print(score)
    print(epr)
    sns.set(rc={'figure.figsize': (12, 8)})
    plt.rcParams.update({'figure.autolayout': True})
    plt.subplot(1, 2, 1)
    sns.heatmap(score, annot=True, fmt=".3f", robust="True")
    plt.title("Confidence Score")
    print(plt.xticks())
    plt.xticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
               ["Cell-type-specific(500)", "Non-specific(500)", "STRING(500)", "Cell-type-specific(1000)",
                "Non-specific(1000)", "STRING(1000)"])
    plt.xticks(rotation=-30, ha="left")
    plt.xlabel("")
    plt.subplots_adjust(wspace=0.25)
    # -------------------------
    plt.subplot(1, 2, 2)
    ax = sns.heatmap(epr, annot=True, fmt=".3f", robust="True", cmap="YlGnBu_r")
    ax.axes.yaxis.set_visible(False)
    plt.title("EPR")
    plt.xticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
               ["Cell-type-specific(500)", "Non-specific(500)", "STRING(500)", "Cell-type-specific(1000)",
                "Non-specific(1000)", "STRING(1000)"])
    plt.xticks(rotation=-30, ha="left")
    plt.xlabel("")
    plt.suptitle(data_name)
    plt.show()


def dream5_TF_TF_eval():
    base_dir = "./output/"
    sub_dir = "DREAM5_alg_output/"
    alg_list = ["GENIE3", "NARROMI", "GRNBOOST2", "PIDC", "ppcor", "PLSNET", "TIGRESS", "Lasso", "Stable Lasso",
                "GENLAB", "ensemble catnet", "catnet", "CLR", "pairwise entropy", "Pearson", "conditional entropy",
                "BMA"]
    res_list = ["e_coli", "in_silico", "s_cere"]
    auroc_res = pd.DataFrame(index=res_list, columns=alg_list)
    aupr_res = auroc_res.copy()
    conf_score_res = auroc_res.copy()
    epr_res = auroc_res.copy()
    for i in tqdm(alg_list):
        for res in res_list:
            print(i)
            print(res)
            pred = pd.read_csv(base_dir + sub_dir + i + "/" + res + ".txt", sep="\t", header=None)
            gs = pd.read_csv("input/" + res + "/" + "gold_standard.tsv", sep="\t", header=None)
            tfs = list(pd.read_csv("input/" + res + "/" + "transcription_factors.tsv", sep="\t", header=None)[0])
            pred = pred[pred[0].isin(tfs) & pred[1].isin(tfs)]
            gs = gs[gs[0].isin(tfs) & gs[1].isin(tfs)]
            pdf_aupr = scio.loadmat("input/" + res + "/" + "AUPR.mat")
            pdf_auroc = scio.loadmat("input/" + res + "/" + "AUROC.mat")
            auroc, aupr, conf_score, epr = eval.run(pred, gs, pdf_aupr, pdf_auroc)
            auroc_res.loc[res, i] = auroc
            aupr_res.loc[res, i] = aupr
            conf_score_res.loc[res, i] = conf_score
            epr_res.loc[res, i] = epr
            print(auroc, aupr, conf_score, epr)
    # ------------------------------------------------
    for j in tqdm(res_list):
        auroc_list = []
        aupr_list = []
        conf_score_list = []
        epr_list = []
        dir = "./output/" + j + "/"
        pred_list = [i for i in os.listdir(dir) if i.startswith("prediction")]
        for pred in tqdm(pred_list):
            pred = pd.read_csv(dir + pred, sep="\t", header=None)
            gs = pd.read_csv("input/" + j + "/" + "gold_standard.tsv", sep="\t", header=None)
            tfs = list(pd.read_csv("input/" + j + "/" + "transcription_factors.tsv", sep="\t", header=None)[0])
            pred = pred[pred[0].isin(tfs) & pred[1].isin(tfs)]
            gs = gs[gs[0].isin(tfs) & gs[1].isin(tfs)]
            pdf_aupr = scio.loadmat("input/" + j + "/" + "AUPR.mat")
            pdf_auroc = scio.loadmat("input/" + j + "/" + "AUROC.mat")
            auroc, aupr, conf_score, epr = eval.run(pred, gs, pdf_aupr, pdf_auroc)
            auroc_list.append(auroc)
            aupr_list.append(aupr)
            conf_score_list.append(conf_score)
            epr_list.append(epr)
        auroc = np.mean(auroc_list)
        aupr = np.mean(aupr_list)
        conf_score = np.mean(conf_score_list)
        epr = np.mean(epr_list)
        auroc_res.loc[j, "Our"] = auroc
        aupr_res.loc[j, "Our"] = aupr
        conf_score_res.loc[j, "Our"] = conf_score
        epr_res.loc[j, "Our"] = epr
        print(auroc, aupr, conf_score, epr)
    print(auroc_res)
    print(aupr_res)
    print(conf_score_res)
    print(epr_res)
    with ExcelWriter("TF_TF_dream5_result.xlsx") as writer:
        auroc_res.to_excel(writer, sheet_name='AUROC', header=True, index=True)
        aupr_res.to_excel(writer, sheet_name='AUPR', header=True, index=True)
        conf_score_res.to_excel(writer, sheet_name='Confidence Score', header=True, index=True)
        epr_res.to_excel(writer, sheet_name='EPR', header=True, index=True)
        writer.save()


def sc_TF_TF_eval():
    base_dir = "./output/"
    sub_dir = "sc_alg_output/"
    alg_list = ["GENIE3", "NARROMI", "GRNBOOST2", "PIDC", "ppcor", "PLSNET", "TIGRESS", "CLR", "Pearson", "Spearman"]
    sc_name = ["hESC", "mDC"]
    gs_type = ["Cell_type_specific", "Non_specific", "STRING"]
    num_list = ["500", "1000"]
    index = [["hESC", "hESC", "hESC", "hESC", "hESC", "hESC", "mDC", "mDC", "mDC", "mDC", "mDC", "mDC"],
             ["Cell_type_specific", "Non_specific", "STRING", "Cell_type_specific", "Non_specific", "STRING",
              "Cell_type_specific", "Non_specific", "STRING", "Cell_type_specific", "Non_specific", "STRING"],
             ["500", "500", "500", "1000", "1000", "1000", "500", "500", "500", "1000", "1000", "1000"]
             ]
    preds = []
    for sc in sc_name:
        for gs in gs_type:
            for num in num_list:
                preds.append([sc, gs, num])
    print(preds)
    auroc_res = pd.DataFrame(index=index, columns=alg_list)
    aupr_res = auroc_res.copy()
    conf_score_res = auroc_res.copy()
    epr_res = auroc_res.copy()
    for alg in alg_list:
        print(alg)
        for pred in tqdm(preds):
            dir = base_dir + sub_dir + alg + "/"
            file = pred[0] + "_" + pred[1] + "_" + pred[2] + ".txt"
            prediction = pd.read_csv(dir + file, sep="\t", header=None)
            gs = pd.read_csv("input/" + pred[0] + "/" + pred[1] + "/" + pred[2] + "/gold_standard.tsv", sep="\t",
                             header=None)
            tfs = list(pd.read_csv("input/" + pred[0] + "/" + pred[1] + "/" + pred[2] + "/transcription_factors.tsv",
                                   sep="\t", header=None)[0])
            prediction = prediction[prediction[0].isin(tfs) & prediction[1].isin(tfs)]
            gs = gs[gs[0].isin(tfs) & gs[1].isin(tfs)]
            pdf_aupr = scio.loadmat("input/" + pred[0] + "/" + pred[1] + "/" + pred[2] + "/AUPR.mat")
            pdf_auroc = scio.loadmat("input/" + pred[0] + "/" + pred[1] + "/" + pred[2] + "/AUROC.mat")
            auroc, aupr, conf_score, epr = eval.run(prediction, gs, pdf_aupr, pdf_auroc)
            auroc_res.loc[[pred], alg] = auroc
            aupr_res.loc[[pred], alg] = aupr
            conf_score_res.loc[[pred], alg] = conf_score
            epr_res.loc[[pred], alg] = epr
    # -------------------------------------------
    for sc in sc_name:
        for gss in gs_type:
            for num in num_list:
                print(sc + gss + num)
                auroc_list = []
                aupr_list = []
                conf_score_list = []
                epr_list = []
                dir = "./input/" + sc + "/" + gss + "/" + num + "/"
                tfs = list(pd.read_csv(dir + "transcription_factors.tsv", sep="\t", header=None)[0])
                pdf_aupr = scio.loadmat(dir + "AUPR.mat")
                pdf_auroc = scio.loadmat(dir + "AUROC.mat")
                files = [i for i in os.listdir("./output/" + sc + "/" + gss + "/" + num + "/") if
                         i.startswith("prediction")]
                for file in tqdm(files):
                    gs = pd.read_csv(dir + "gold_standard.tsv", sep="\t", header=None)
                    pred = pd.read_csv("./output/" + sc + "/" + gss + "/" + num + "/" + file, sep="\t", header=None)
                    # print(pred)
                    pred = pred[pred[0].isin(tfs) & pred[1].isin(tfs)]
                    gs = gs[gs[0].isin(tfs) & gs[1].isin(tfs)]
                    auroc, aupr, conf_score, epr = eval.run(pred, gs, pdf_aupr, pdf_auroc)
                    auroc_list.append(auroc)
                    aupr_list.append(aupr)
                    conf_score_list.append(conf_score)
                    epr_list.append(epr)
                auroc = np.mean(auroc_list)
                aupr = np.mean(aupr_list)
                conf_score = np.mean(conf_score_list)
                epr = np.mean(epr_list)
                auroc_res.loc[[[sc, gss, num]], "Our"] = auroc
                aupr_res.loc[[[sc, gss, num]], "Our"] = aupr
                conf_score_res.loc[[[sc, gss, num]], "Our"] = conf_score
                epr_res.loc[[[sc, gss, num]], "Our"] = epr
    print(auroc_res)
    print(aupr_res)
    print(conf_score_res)
    print(epr_res)
    with ExcelWriter("TF_TF_sc_result.xlsx") as writer:
        auroc_res.to_excel(writer, sheet_name='AUROC', header=True, index=True)
        aupr_res.to_excel(writer, sheet_name='AUPR', header=True, index=True)
        conf_score_res.to_excel(writer, sheet_name='Confidence Score', header=True, index=True)
        epr_res.to_excel(writer, sheet_name='EPR', header=True, index=True)
        writer.save()


def draw_para_heatmap():
    data = pd.read_excel("E:\\sync\\OneDrive - for personal\\wzwzwzwz\\res_for_chart\\in_silico.xlsx",
                         sheet_name="cor_th=0.35", header=0, index_col=0)
    print(data)
    data = (data - data.mean()) / (data.std())
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="darkgrid", font_scale=1.3)
    # plt.ticklabel_format(axis="both", style='sci', useMathText=True)
    sns.lineplot(data=data)

    plt.show()


if __name__ == '__main__':
    # draw_dream5_res2("AUROC", [0.45, 0.85])
    # draw_dream5_res2("AUPR", [0, 0.35])
    # draw_dream5_res("AUPR", [0, 0.4])
    # draw_heatmap_for_dream5()
    # draw_sc_res_auroc("mDC", "AUROC", [0.45, 0.65])
    draw_sc_res_aupr("mDC", "AUPR", [0, 0.05])
    # draw_sc_res("mDC", "AUROC", [0.4, 0.65])
    # draw_sc_res("hESC", "AUPR", [0, 0.30])
    # draw_sc_res("mDC", "AUPR", [0, 0.25])
    # draw_heatmap_for_sc("hESC")
    # draw_heatmap_for_sc("mDC")
    # eval_ANM()

    # draw_para_heatmap()
