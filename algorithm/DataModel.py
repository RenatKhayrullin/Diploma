from __future__ import print_function
import io
import numpy as np
from scipy.sparse import *
from sklearn.preprocessing import normalize
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from Algorithm import clustype_no_clus_algorithm
from Algorithm import clustype_algorithm

np.random.seed(2018)


def normalize_graph(G):
    G_col = G.sum(axis=0)
    G_row = G.sum(axis=1)

    miss_col = 0
    col_val = []
    for i in range(G_col.shape[1]):
        val = G_col[0, i]
        if val > 0:
            col_val.append(pow(val, -0.5))
        else:
            col_val.append(0.0)
            miss_col += 1

    miss_row = 0
    row_val = []
    for i in range(G_row.shape[0]):
        val = G_row[i, 0]
        if val > 0:
            row_val.append(pow(val, -0.5))
        else:
            row_val.append(0.0)
            miss_row += 1

    G = diags(row_val, 0) * G * diags(col_val, 0)
    return G


def load_graph(filename, row_shape, col_shape):
    rows = []
    cols = []
    vals = []
    with io.open(filename, 'r', encoding='utf8') as G:
        for line in G:
            line = line.replace("\n", "").split(" ")
            rows.append(int(line[0]))
            cols.append(int(line[1]))
            vals.append(float(line[2]))
        G.close()
    return coo_matrix((vals, (rows, cols)), shape=(row_shape, col_shape)).tocsr()


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    f.close()
    return i + 1


def create_Y_0(Y, rand_size):
    row_size, col_size = Y.shape
    # print "Y shape: " + str(row_size) + " " + str(col_size)
    count_list = Y.sum(axis=0).tolist()[0]
    type_counts = zip(entity_types_list, count_list)
    print(type_counts)

    pers_rows = []  # 0
    loc_rows = []  # 1
    org_rows = []  # 2

    for i in xrange(row_size):
        if Y[i, 0] == 1:
            pers_rows.append(i)
        if Y[i, 1] == 1:
            loc_rows.append(i)
        if Y[i, 2] == 1:
            org_rows.append(i)

    pers_rand_size = int(type_counts[0][1] * rand_size / 100)
    loc_rand_size = int(type_counts[1][1] * rand_size / 100)
    org_rand_size = int(type_counts[2][1] * rand_size / 100)

    rand_pers_rows = np.random.choice(np.asarray(pers_rows), pers_rand_size, replace=False)
    rand_loc_rows = np.random.choice(np.asarray(loc_rows), loc_rand_size, replace=False)
    rand_org_rows = np.random.choice(np.asarray(org_rows), org_rand_size, replace=False)

    Y_0_rows = []
    Y_0_cols = []
    Y_0_vals = []

    for idx in rand_pers_rows:
        Y_0_rows.append(idx)
        Y_0_cols.append(0)
        Y_0_vals.append(1)

    for idx in rand_loc_rows:
        Y_0_rows.append(idx)
        Y_0_cols.append(1)
        Y_0_vals.append(1)

    for idx in rand_org_rows:
        Y_0_rows.append(idx)
        Y_0_cols.append(2)
        Y_0_vals.append(1)

    return coo_matrix((Y_0_vals, (Y_0_rows, Y_0_cols)), shape=(row_size, col_size)).tocsr(), \
           list(set(xrange(row_size)) - set(Y_0_rows)), \
           Y_0_rows


def compute_score(Y, score_Y, test_rows):
    # Evaluation
    # http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
    # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score
    # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score
    # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    test_Y = Y[test_rows, :].todense()
    answers = score_Y[test_rows, :].argmax(axis=1)
    score_Y = score_Y[test_rows, :]

    for i in xrange(len(test_rows)):
        answer_class = answers[i, 0]
        score_Y[i, :] = 0
        score_Y[i, answer_class] = 1

    precision = dict()
    recall = dict()
    f1_measure = dict()
    for i in range(3):
        precision[i] = precision_score(test_Y[:, i], score_Y[:, i], average="binary")
        recall[i] = recall_score(test_Y[:, i], score_Y[:, i], average="binary")
        f1_measure[i] = f1_score(test_Y[:, i], score_Y[:, i], average="binary")
        print(str(i) + " Precision = " + str(precision[i]) + " Recall = " + str(recall[i]) + " F1_measure = " + str(f1_measure[i]))


if __name__ == "__main__":
    dicts_path = "/Users/Reist/PycharmProjects/Diploma/processed_text/"
    graphs_path = "/Users/Reist/PycharmProjects/Diploma/graphs_constructor/"

    entity_types_list = []
    with io.open(graphs_path + "entity_types.txt", 'r', encoding='utf8') as entity_types:
        for i, line in enumerate(entity_types):
            entity_types_list.append(line.replace("\n", ""))
        entity_types.close()

    em_size = file_len(dicts_path + "entity_mentions.txt")
    em_dict_size = file_len(dicts_path + "entity_dictionary.txt")
    rp_dict_size = file_len(dicts_path + "relation_dictionary.txt")

    rp_context_w_size = file_len(graphs_path + "rp_context_words_dict.txt")
    rp_inner_w_size = file_len(graphs_path + "rp_words_dict.txt")

    types_size = file_len(graphs_path + "entity_types.txt")

    G_Q = load_graph(graphs_path + "G_Q.txt", em_size, em_dict_size)    # entity_mentions_size X mentions_dict_size
    G_Q = normalize(G_Q, norm='l2', axis=0)
    G_L = load_graph(graphs_path + "G_L.txt", em_size, rp_dict_size)    # entity_mentions_size X relations_dict_size
    G_L = normalize(G_L, norm='l2', axis=0)
    S_L = normalize_graph(G_Q.T * G_L)
    G_R = load_graph(graphs_path + "G_R.txt", em_size, rp_dict_size)    # entity_mentions_size X relations_dict_size
    G_R = normalize(G_R, norm='l2', axis=0)
    S_R = normalize_graph(G_Q.T * G_R)
    G_M = load_graph(graphs_path + "G_M.txt", em_size, em_size)         # entity_mentions_size X entity_mentions_size
    G_M = normalize(G_M, norm='l2', axis=0)
    S_M = normalize_graph(G_M)
    F_s = load_graph(graphs_path + "F_s.txt", rp_dict_size, rp_context_w_size)
    F_c = load_graph(graphs_path + "F_c.txt", rp_dict_size, rp_inner_w_size)

    Y = load_graph(graphs_path + "Y.txt", em_size, types_size)


    tol = 5e-4
    gamma, mu, alpha = 0.5, 0.5, 1.

    initial_markup_size = 1
    glob_iter_count, inner_iter_count, nmf_iter_count = 100, 100, 100
    T, K = 3, 100

    Y_0, test_rows, train_rows = create_Y_0(Y, initial_markup_size)
    score_Y, C, P_L, P_R = clustype_no_clus_algorithm(S_L, S_R, S_M, G_Q, G_L, G_R, Y_0, gamma, mu, glob_iter_count)
    print("WITHOUT CLUSTERING:")
    compute_score(Y, score_Y, test_rows)

    score_Y = clustype_algorithm(S_L, S_R, S_M, G_Q, G_L, G_R,
                                 Y_0, F_s, F_c, score_Y, C, P_L, P_R,
                                 gamma, mu, alpha, T, K, glob_iter_count, inner_iter_count, nmf_iter_count, tol)
    print("WITH CLUSTERING:")
    compute_score(Y, score_Y, test_rows)
