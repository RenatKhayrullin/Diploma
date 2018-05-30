import io
import numpy as np
from scipy.sparse import *
from sklearn.preprocessing import normalize
from Algorithm import create_initial_matrices

np.random.seed(2018)


def normalize_graph(G):
    # print 'row-by-col = ', G.shape
    G_col = G.sum(axis=0)  # 1-by-#col vector
    G_row = G.sum(axis=1)  # #row-by-1 vector

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
    # print miss_row, 'rows missed', miss_col, 'col missed'

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


def create_Y_0(Y):
    row_size, col_size = Y.shape
    # print "Y shape: " + str(row_size) + " " + str(col_size)
    count_list = Y.sum(axis=0).tolist()[0]
    type_counts = zip(entity_types_list, count_list)
    # print type_counts

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

    pers_rand_size = int(type_counts[0][1] * 7 / 100)
    loc_rand_size = int(type_counts[1][1] * 7 / 100)
    org_rand_size = int(type_counts[2][1] * 7 / 100)

    # print "INITIAL SAMPLES COUNT: " + "PERS = " + str(pers_rand_size) + " LOC = " + str(loc_rand_size) + " ORG = " + str(org_rand_size)

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

    return coo_matrix((Y_0_vals, (Y_0_rows, Y_0_cols)), shape=(row_size, col_size)).tocsr()


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
    np.save("G_Q.npy", G_Q)
    print "G_Q shape: " + str(G_Q.shape)
    print "G_Q computed and saved"

    G_L = load_graph(graphs_path + "G_L.txt", em_size, rp_dict_size)    # entity_mentions_size X relations_dict_size
    G_L = normalize(G_L, norm='l2', axis=0)
    np.save("G_L.npy", G_L)
    print "G_L shape: "+str(G_L.shape)
    print "G_L computed and saved"

    S_L = normalize_graph(G_Q.T * G_L)
    np.save("S_L.npy", S_L)
    print "S_L shape: " + str(S_L.shape)
    print "S_L computed and saved"

    G_R = load_graph(graphs_path + "G_R.txt", em_size, rp_dict_size)    # entity_mentions_size X relations_dict_size
    G_R = normalize(G_R, norm='l2', axis=0)
    np.save("G_R.npy", G_R)
    print "G_R shape: " + str(G_R.shape)
    print "G_R computed and saved"

    S_R = normalize_graph(G_Q.T * G_R)
    np.save("S_R.npy", S_R)
    print "S_R shape: " + str(S_R.shape)
    print "S_R computed and saved"

    G_M = load_graph(graphs_path + "G_M.txt", em_size, em_size)         # entity_mentions_size X entity_mentions_size
    G_M = normalize(G_M, norm='l2', axis=0)
    np.save("G_M.npy", G_M)
    print "G_M shape: " + str(G_M.shape)
    print "G_M computed and saved"

    S_M = normalize_graph(G_M)
    np.save("S_M.npy", S_M)
    print "S_M shape: " + str(S_M.shape)
    print "S_M computed and saved"

    F_s = load_graph(graphs_path + "F_s.txt", rp_dict_size, rp_context_w_size)
    np.save("F_s.npy", F_s)
    print "F_s shape: " + str(F_s.shape)
    print "F_s computed and saved"

    F_c = load_graph(graphs_path + "F_c.txt", rp_dict_size, rp_inner_w_size)
    np.save("F_c.npy", F_c)
    print "F_c shape: " + str(F_c.shape)
    print "F_c computed and saved"

    Y = load_graph(graphs_path + "Y.txt", em_size, types_size)
    Y_0 = create_Y_0(Y)
    np.save("Y_0.npy", Y_0)
    print "Y_0 shape: " + str(Y_0.shape)
    print "Y_0 computed and saved"

    gamma = 0.5
    mu = 0.5
    iter_count = 20
    T = 3

    Y, C, P_L, P_R = create_initial_matrices(S_L, S_R, S_M, G_Q, G_L, G_R, Y_0, gamma, mu, T, iter_count)