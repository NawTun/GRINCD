import pandas as pd
import numpy as np
import os
from anm import ANM
from tqdm.auto import tqdm
import multiprocessing
from multiprocessing import Process

# from tensorflow.python.client import device_lib

hyper_parameters = {
    'number_of_walks': 1,
    'length': 5,
    'batch_size': 50,
    'epochs': 4,
    'bins': 10,
    "mi_th": 0.3,
    "cor_th": 0.3,
}


# def get_gpus():
#     local_device_protos = device_lib.list_local_devices()
#     GPU_list = []
#     for device in local_device_protos:
#         if device.device_type == "GPU":
#             GPU_list.append(device)
#     GPU_list.sort(key=lambda GPU: GPU.memory_limit, reverse=True)
#     return GPU_list


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


def mi_net(tf_num, data_file, threshold):
    dir = data_file[:-19]
    expr = pd.read_csv(data_file, sep='\t', header=0)
    expr = (expr - expr.mean()) / (expr.std())
    mi_net_file = dir + "/mi.tsv"
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
    return mi_mtx


def ensemble(pred, alpha=0.5):
    pred1 = pred[0].sort_values(by=['tf', 'tg'], ascending=True)
    pred2 = pred[1].sort_values(by=['tf', 'tg'], ascending=True)
    en_res = pd.DataFrame(columns=['tf', 'tg', 'conf'])
    en_res['tf'] = pred1['tf']
    en_res['tg'] = pred1['tg']
    en_res['conf'] = pred1['conf'] * alpha + pred2['conf'] * (1 - alpha)
    en_res.sort_values(by='conf', ascending=False)
    en_res = en_res.loc[:, ['tf', 'tg']]
    return en_res


def co_expr_net(tf_num, data_file, threshold):
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


def from_pandas_to_stell(adj_mtx, data_file):
    expr = pd.read_csv(data_file, sep='\t', header=0)
    expr = (expr - expr.mean()) / (expr.std())
    node_features = expr.T

    genes = adj_mtx.columns
    edges = np.where(adj_mtx.values == 1)
    index, columns = genes[edges[0]], genes[edges[1]]
    pd_edges = pd.DataFrame(
        {"source": index, "target": columns}
    )
    # G = StellarGraph(node_features, pd_edges)
    G = (node_features, pd_edges)
    # print(G.info())
    return G


def node_embedding(ori_G, gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    from stellargraph import StellarGraph
    from stellargraph.data import UnsupervisedSampler
    from stellargraph.mapper import GraphSAGELinkGenerator
    from stellargraph.layer import GraphSAGE, link_classification
    from tensorflow import keras
    from stellargraph.mapper import GraphSAGENodeGenerator
    G = StellarGraph(ori_G[0], ori_G[1])
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


def calc_dirction(pd_node_embeddings, tfs):
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
    pred = pd.DataFrame({'tf': tf_list, 'tg': tg_list, 'conf': conf_list})
    pred = pred.sort_values(by='conf', ascending=False)
    return pred


def process_pipeline(tag, tfs, exp_file, queue, gpu_id):
    if tag is "corr":
        mtx = co_expr_net(tf_num=len(tfs), data_file=exp_file, threshold=hyper_parameters['cor_th'])
    elif tag is "mi":
        mtx = mi_net(tf_num=len(tfs), data_file=exp_file, threshold=hyper_parameters['mi_th'])
    else:
        raise Exception("pls select the right type of relation network")
    G = from_pandas_to_stell(mtx, exp_file)
    embedding = node_embedding(G, gpu_id)
    pred = calc_dirction(embedding, tfs=tfs)
    queue.put({tag: pred})


if __name__ == '__main__':
    net_name = "in_silico"
    input_base_dir = "/home/fengke/pycode/my_exp/paper/exp/input/"
    output_base_dir = "/home/fengke/pycode/my_exp/paper/exp/output/"
    if net_name in ["in_silico", "e_coli", "s_cere"]:
        tfs = list(pd.read_csv(input_base_dir + net_name + "/transcription_factors.tsv", sep="\t", header=None)[0])
        tf_num = len(tfs)
        exp_file = input_base_dir + net_name + "/expression_data.tsv"
        output_file = output_base_dir + net_name + "/prediction.txt"
    elif net_name in ["hESC", "mDC"]:
        gs_type = "Cell_type_specific/"
        num_type = "500/"
        tfs = list(
            pd.read_csv(input_base_dir + net_name + "/" + gs_type + num_type + "/transcription_factors.tsv", sep="\t",
                        header=None)[0])
        tf_num = len(tfs)
        exp_file = input_base_dir + net_name + "/" + gs_type + num_type + "/expression_data.tsv"
        output_file = output_base_dir + net_name + "/" + gs_type + num_type + "/prediction.txt"
    else:
        raise Exception("No such dataset as " + net_name)
    # GPU_list = get_gpus()
    # print(GPU_list)
    queue = multiprocessing.Queue()
    p_corr = Process(target=process_pipeline, args=("corr", tfs, exp_file, queue, "2"))  # 实例化进程对象
    p_corr.start()
    p_mi = Process(target=process_pipeline, args=("mi", tfs, exp_file, queue, "3"))  # 实例化进程对象
    p_mi.start()
    p_corr.join()
    p_mi.join()
    pred_result = [queue.get() for _ in range(2)]
    # corr_mtx = co_expr_net(tf_num=tf_num, data_file=exp_file)
    # corr_G = from_pandas_to_stell(corr_mtx)
    # embedding = node_embedding(corr_G)
    # corr_pred = calc_dirction(embedding, tfs=tfs)
    # # print(pred1)
    # # run(predictions=[pred1, ], network_list=(1,), required_files=False)
    #
    # mi_mtx = mi_net(tf_num=tf_num, data_file=exp_file)
    # mi_G = from_pandas_to_stell(mi_mtx)
    # embedding = node_embedding(mi_G)
    # mi_pred = calc_dirction(embedding, tfs=tfs)
    # print(pred2)
    # run(predictions=[pred2, ], network_list=(1,), required_files=False)
    pred_list = []
    for i in pred_result:
        if "corr" in i.keys():
            pred_list.append(i["corr"])
        elif "mi" in i.keys():
            pred_list.append(i["mi"])
    pred = ensemble(pred_list)
    pred.to_csv(output_file, sep="\t", header=False, index=False)
