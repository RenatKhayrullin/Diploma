import io
from collections import defaultdict
from operator import itemgetter
from math import log, sqrt
from string import punctuation
from scipy.sparse import *
from scipy import *
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize
from numpy import *

graph_path = "/Users/Reist/PycharmProjects/prepate_data/graph_constructor/"

class DataModel:
    def __init__(self):
        G_Q = None  # entity_mentions_size X mentions_dict_size
        G_L = None  # entity_mentions_size X relations_dict_size
        G_R = None  # entity_mentions_size X relations_dict_size
        G_M = None  # entity_mentions_size X entity_mentions_size
        F_s = None  # relations_dict_size X unique_context_words_size
        F_c = None  # relations_dict_size X unique_inner_rp_words_size

    @staticmethod
    def normalize_graph(G):
        pass

    def load_graph(self, filename, row_shape, col_shape):
        rows = []
        cols = []
        vals = []
        with io.open(graph_path + filename, 'r', encoding='utf8') as G:
            for line in G:
                line = line.replace("\n", "").split(" ")
                rows.append(int(line[0]))
                cols.append(int(line[1]))
                vals.append(int(line[2]))
            G.close()
        return coo_matrix((vals, (rows, cols)), shape=(row_shape, col_shape)).tocsr()