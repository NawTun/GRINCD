from tqdm.auto import tqdm
import pandas as pd
import itertools
from sklearn.utils import shuffle
import eval


def modify_exp(df, tfs, dir):
    """
    Divide the expression matrix into a transcription factor part and a target gene part.
    :param df: Expression matrix
    :param tfs: The list of transcription factors.
    :param dir: Directory of output.
    :return: None
    """
    print(tfs)
    all_genes = set(df.columns)
    other_genes = all_genes.difference(set(tfs[0]))
    first_half = df.loc[:, list(tfs[0])]
    second_half = df.loc[:, other_genes]
    new_df = pd.concat([first_half, second_half], axis=1)
    print(new_df)
    new_df.to_csv(dir + "expression_data.tsv", sep="\t", header=True, index=False)


def mapping_id(dir):
    """
    Associate the gene id with the gene name.
    :param dir: Directory of expression matrix.
    :return: None
    """
    df = pd.read_csv(dir + "expression_data.tsv", sep="\t", header=0)
    no_of_genes = df.shape[1]
    gene_ids = ["G" + str(i + 1) for i in range(no_of_genes)]
    gene_mapping = pd.DataFrame(columns=["#ID", "Name"])
    gene_mapping["#ID"] = gene_ids
    gene_mapping["Name"] = df.columns
    print(gene_mapping)
    gene_mapping.to_csv(dir + "gene_ids.tsv", sep="\t", header=True, index=False)
    df.columns = gene_mapping["#ID"]
    df.to_csv(dir + "expression_data.tsv", sep="\t", header=True, index=False)


def mapping_gs(tfs, gs, dir):
    """
    Associate the gene id with the gene name in gold standard.
    :param tfs: The list of transcription factors.
    :param gs: Gold standard.
    :param dir: Directory of files.
    :return: 
    """
    gene_mapping = pd.read_csv(dir + "gene_ids.tsv", sep="\t", header=0)
    print(tfs)
    mapping_dict = gene_mapping.set_index(["Name"])["#ID"].to_dict()

    tfs[0] = tfs[0].map(mapping_dict)
    gs[0] = gs[0].map(mapping_dict)
    gs[1] = gs[1].map(mapping_dict)
    print(tfs)
    tfs.to_csv(dir + "transcription_factors.tsv", sep="\t", header=False, index=False)
    gs.to_csv(dir + "gold_standard.tsv", sep="\t", header=False, index=False)


def main():
    input_dir = './input/'
    a = ["hESC", "mDC"]
    b = ["Cell_type_specific", "Non_specific", "STRING"]
    c = ["500", "1000"]
    dir_list = []
    for i in a:
        for j in b:
            for k in c:
                path = input_dir + i + "/" + j + "/" + k + "/"
                dir_list.append(path)
    for dir in tqdm(dir_list):
        df = pd.read_csv(dir + "expression_data.tsv", sep="\t", header=0)
        gs = pd.read_csv(dir + "gold_standard.tsv", sep="\t", header=None)
        tfs = pd.read_csv(dir + "transcription_factors.tsv", sep="\t", header=None)
        tfs = list(tfs[0])
        genes = list(df.columns)
        new = itertools.product(tfs, genes)
        new = pd.DataFrame(new)
        auroc_list = []
        aupr_list = []
        net_name = dir.split("/")[0]
        print(net_name)
        for _ in tqdm(range(100000)):
            random_edge_list = shuffle(new)
            auroc, aupr, conf_score = eval.run(net_name, prediction=random_edge_list, gold_standard=gs.copy(),
                                               required_file=False)
            print(auroc, aupr)
            auroc_list.append(auroc)
            aupr_list.append(aupr)
        ret = pd.DataFrame(columns=["auroc", "aupr"])
        ret["auroc"] = auroc_list
        ret["aupr"] = aupr_list
        # print(ret)
        ret.to_csv(dir + "random_eval_res.tsv", sep="\t", header=True, index=False)

        # modify_exp(df, tfs, dir)
        # mapping_id(dir)
        # mapping_gs(tfs, gs, dir)


if __name__ == '__main__':
    main()
