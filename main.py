import time
import pandas as pd
import numpy as np
from stellargraph import StellarGraph
from stellargraph.data import UnsupervisedSampler
from stellargraph.mapper import GraphSAGELinkGenerator
from stellargraph.layer import GraphSAGE, link_classification
from tensorflow import keras
from stellargraph.mapper import GraphSAGENodeGenerator
from anm import ANM
from tqdm.auto import tqdm

hyper_parameters = {
    'number_of_walks': 1,
    'length': 5,
    'batch_size': 50,
    'epochs': 4,
    'bins': 10
}


def calc_MI(X, Y, bins):
    XY_hist = np.histogram2d(X, Y, bins)[0]
    X_hist = np.histogram(X, bins)[0]
    Y_hist = np.histogram(Y, bins)[0]
    X_entropy = calc_entropy(X_hist)
    Y_entropy = calc_entropy(Y_hist)
    XY_entropy = calc_entropy(XY_hist)
    MI = X_entropy + Y_entropy - XY_entropy
    return MI


def calc_entropy(c):
    c_prob = c / float(np.sum(c))
    c_prob = c_prob[np.nonzero(c_prob)]
    H = -sum(c_prob * np.log2(c_prob))
    return H


def mi_net(tf_num, data_file, threshold=0.3, ):
    expr = pd.read_csv(data_file, sep='\t', header=0)
    expr = (expr - expr.mean()) / (expr.std())
    mi_mtx = np.zeros((expr.shape[1], expr.shape[1]))
    for tf in tqdm(range(tf_num)):
        for tg in range(expr.shape[1]):
            A = expr.iloc[:, tf].values
            B = expr.iloc[:, tg].values
            result_NMI = calc_MI(A, B, bins=10)
            mi_mtx[tf, tg] = result_NMI
    mi_mtx = np.triu(mi_mtx, 0)
    mi_mtx = mi_mtx + mi_mtx.T
    np.fill_diagonal(mi_mtx, 0)
    mi_mtx[np.where(abs(mi_mtx) > threshold)] = 1
    mi_mtx[np.where(abs(mi_mtx) <= threshold)] = 0
    mi_mtx = pd.DataFrame(mi_mtx, index=expr.columns, columns=expr.columns, dtype=int)
    print(mi_mtx)
    return mi_mtx

# ensenmble of linear pipeline and non-linear pipeline
def ensemble(pred1, pred2, alpha=0.5):
    pred1 = pred1.sort_values(by=['tf', 'tg'], ascending=True)
    pred2 = pred2.sort_values(by=['tf', 'tg'], ascending=True)
    en_res = pd.DataFrame(columns=['tf', 'tg', 'conf'])
    en_res['tf'] = pred1['tf']
    en_res['tg'] = pred1['tg']
    en_res['conf'] = pred1['conf'] * alpha + pred2['conf'] * (1 - alpha)
    en_res.sort_values(by='conf', ascending=False)
    en_res = en_res.loc[:, ['tf', 'tg']]
    return en_res


# calculate correlation matrix
def co_expr_net(tf_num, data_file, threshold=0.3):
    expr = pd.read_csv(data_file, sep='\t', header=0)
    expr = (expr - expr.mean()) / (expr.std())
    corr_mtx = expr.corr()
    corr_mtx.iloc[tf_num:, tf_num:] = 0
    np.fill_diagonal(corr_mtx.values, 0)
    np_corr_mtx = corr_mtx.values
    np_corr_mtx[np.where(abs(np_corr_mtx) > threshold)] = 1
    np_corr_mtx[np.where(abs(np_corr_mtx) <= threshold)] = 0
    np_corr_mtx = np_corr_mtx.astype(np.int)
    corr_mtx = pd.DataFrame(np_corr_mtx, index=corr_mtx.index, columns=corr_mtx.columns)
    return corr_mtx


# construct stellargraph
def from_pandas_to_stell(adj_mtx, data_file='input/in_silico/expression_data.tsv'):
    expr = pd.read_csv(data_file, sep='\t', header=0)
    expr = (expr - expr.mean()) / (expr.std())
    node_features = expr.T

    genes = adj_mtx.columns
    edges = np.where(adj_mtx.values == 1)
    index, columns = genes[edges[0]], genes[edges[1]]
    pd_edges = pd.DataFrame(
        {"source": index, "target": columns}
    )
    G = StellarGraph(node_features, pd_edges)
    print(G.info())
    return G


# gene representation
def node_embedding(G):
    nodes = list(G.nodes())
    number_of_walks = hyper_parameters['number_of_walks']
    length = hyper_parameters['length']
    unsupervised_samples = UnsupervisedSampler(
        G, nodes=nodes, length=length, number_of_walks=number_of_walks
    )
    batch_size = hyper_parameters['batch_size']
    epochs = hyper_parameters['epochs']
    num_samples = [10, 5]
    generator = GraphSAGELinkGenerator(G, batch_size, num_samples)
    train_gen = generator.flow(unsupervised_samples)
    layer_sizes = [50, 50]
    graphsage = GraphSAGE(
        layer_sizes=layer_sizes, generator=generator, bias=True, dropout=0.0, normalize="l2"
    )
    x_inp, x_out = graphsage.in_out_tensors()
    prediction = link_classification(
        output_dim=1, output_act="sigmoid", edge_embedding_method="ip"
    )(x_out)
    model = keras.Model(inputs=x_inp, outputs=prediction)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.binary_crossentropy,
        metrics=[keras.metrics.binary_accuracy],
    )
    history = model.fit(
        train_gen,
        epochs=epochs,
        verbose=1,
        use_multiprocessing=False,
        workers=4,
        shuffle=True,
    )
    x_inp_src = x_inp[0::2]
    x_out_src = x_out[0]
    embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)
    node_gen = GraphSAGENodeGenerator(G, batch_size, num_samples).flow(nodes)
    node_embeddings = embedding_model.predict(node_gen, workers=4, verbose=1)
    pd_node_embeddings = pd.DataFrame(node_embeddings.T, columns=nodes)
    return pd_node_embeddings

# calculate importance of regulatory relations using ANM
def calc_dirction(pd_node_embeddings, tfs):
    t1 = time.time()
    tf_list = []
    tg_list = []
    conf_list = []
    anm = ANM()
    for tf in tqdm(tfs):
        for tg in pd_node_embeddings.columns:
            if tf == tg:
                continue
            tf_list.append(tf)
            tg_list.append(tg)
            score = anm.predict_proba((pd_node_embeddings[tf].values, pd_node_embeddings[tg].values))
            conf_list.append(score)
    t2 = time.time()
    print(t2 - t1)
    pred = pd.DataFrame({'tf': tf_list, 'tg': tg_list, 'conf': conf_list})
    pred = pred.sort_values(by='conf', ascending=False)
    return pred


if __name__ == '__main__':
    input_base_dir = "/home/fengke/pycode/my_exp/paper/exp/input/"
    output_base_dir = "/home/fengke/pycode/my_exp/paper/exp/output/"
    net_name = "in_silico"
    tfs = list(pd.read_csv(input_base_dir + net_name + "/" + "transcription_factors.tsv", sep="\t", header=None)[0])
    tf_num = len(tfs)
    corr_mtx = co_expr_net(tf_num=tf_num, data_file=input_base_dir + net_name + "/" + "expression_data.tsv")
    corr_G = from_pandas_to_stell(corr_mtx)
    embedding = node_embedding(corr_G)
    corr_pred = calc_dirction(embedding, tfs=tfs)
    # print(pred1)
    # run(predictions=[pred1, ], network_list=(1,), required_files=False)

    mi_mtx = mi_net(tf_num=tf_num, data_file=input_base_dir + net_name + "/" + "expression_data.tsv")
    mi_G = from_pandas_to_stell(corr_mtx)
    embedding = node_embedding(mi_G)
    mi_pred = calc_dirction(embedding, tfs=tfs)
    # print(pred2)
    # run(predictions=[pred2, ], network_list=(1,), required_files=False)

    pred = ensemble(corr_pred, mi_pred)
    pred.to_csv(output_base_dir + net_name + "prediction.txt", sep="\t", header=False, index=False)
    # run(predictions=[pred, ], network_list=(1,), required_files=False)
