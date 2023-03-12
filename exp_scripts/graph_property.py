import networkx as nx
import pandas as pd


def main():
    base_dir = "./input/"
    data_list = ["in_silico", "e_coli", "s_cere", "hESC", "mDC"]
    for i in data_list:
        if i in ["in_silico", "e_coli", "s_cere"]:
            gs = pd.read_csv(base_dir + i + "/gold_standard.tsv", sep="\t", header=None)
            gene_df = pd.read_csv(base_dir + i + "/gene_ids.tsv", sep="\t", header=0)
            exp_df = pd.read_csv(base_dir + i + "/expression_data.tsv", sep="\t", header=0)
            genes_len = gene_df.shape[0]
            tfs_len = pd.read_csv(base_dir + i + "/transcription_factors.tsv", sep="\t", header=None).shape[0]
            G = nx.DiGraph()
            t_tfs = list(gs[0])
            t_genes = list(gs[1])
            G.add_nodes_from(t_tfs)
            G.add_nodes_from(t_genes)
            edges = zip(t_tfs, t_genes)
            G.add_edges_from(edges)
            iso_tfs = tfs_len - len(set(t_tfs))
            iso_genes = genes_len - len(set(t_genes))
            print(i)
            print("# samples:", exp_df.shape[0])
            print("# genes:", genes_len)
            print("# tfs:", tfs_len)
            print("# genes that are not regulated by others:", iso_genes)
            print("# tfs that do not regulate others:", iso_tfs)
            print(nx.info(G))
            print("-------------------------------------------------------------")
        else:
            for j in ["Cell_type_specific", "Non_specific", "STRING"]:
                for k in ["500", "1000"]:
                    gs = pd.read_csv(base_dir + i + "/" + j + "/" + k + "/gold_standard.tsv", sep="\t", header=None)
                    gene_df = pd.read_csv(base_dir + i + "/" + j + "/" + k + "/gene_ids.tsv", sep="\t", header=0)
                    exp_df = pd.read_csv(base_dir + i + "/" + j + "/" + k + "/expression_data.tsv", sep="\t", header=0)
                    genes_len = gene_df.shape[0]
                    tfs_len = \
                        pd.read_csv(base_dir + i + "/" + j + "/" + k + "/transcription_factors.tsv", sep="\t",
                                    header=None).shape[0]
                    G = nx.DiGraph()
                    t_tfs = list(gs[0])
                    t_genes = list(gs[1])
                    G.add_nodes_from(t_tfs)
                    G.add_nodes_from(t_genes)
                    edges = zip(t_tfs, t_genes)
                    G.add_edges_from(edges)
                    iso_tfs = tfs_len - len(set(t_tfs))
                    iso_genes = genes_len - len(set(t_genes))
                    print(i + "/" + j + "/" + k)
                    print("# samples:", exp_df.shape[0])
                    print("# genes:", genes_len)
                    print("# tfs:", tfs_len)
                    print("# genes that are not regulated by others:", iso_genes)
                    print("# tfs that do not regulate others:", iso_tfs)
                    print(nx.info(G))
                    print("-------------------------------------------------------------")


if __name__ == '__main__':
    """
    Calculate graph properties of each gold standard.
    """
    main()
