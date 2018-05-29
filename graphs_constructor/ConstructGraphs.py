# coding=utf-8
import io
import sys
import csv
import logging
import warnings
import time
from collections import Counter
from operator import itemgetter
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize
from math import log, exp
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.feature_extraction import text as TextFeatures
from gensim.models import Word2Vec

warnings.simplefilter(action='ignore', category=FutureWarning)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

reload(sys)
sys.setdefaultencoding('utf-8')


def hasNumbers(input_string):
    return any(char.isdigit() for char in input_string)


def word2vec_vocabulary(text_corpus):
    splitted_text_corpus = []
    for doc in text_corpus:
        splitted_text_corpus.append(doc.split(" "))
    word_2_vec_model = Word2Vec(splitted_text_corpus, size=100, window=5, workers=2, negative=5, min_count=1)
    word_2_vec_model.save("word_2_vec.bin")
    return word_2_vec_model


def tf_idf_vocabulary(text_corpus):
    vocabulary = {}
    unigram_model = TextFeatures.TfidfVectorizer(analyzer='word', norm='l2', sublinear_tf=True, smooth_idf=True)
    X = unigram_model.fit_transform(text_corpus)
    word_features = unigram_model.get_feature_names()
    with io.open("vocabulary_tf_idf.txt", "wb") as tf_idf_vocabulary:
        writer = csv.writer(tf_idf_vocabulary)
        for word, tf_idf in sorted(zip(word_features, np.asarray(X.sum(axis=0)).ravel()), key=lambda t: t[1] * -1):
            vocabulary[word] = tf_idf
            writer.writerow([word, tf_idf])
    tf_idf_vocabulary.close()
    return vocabulary


def construct_biadjacency_G(mentions, dictionary, filename):
    with io.open(filename, 'a', encoding='utf8') as G:
        for mention in mentions:
            if mention[1] != "None":
                row = mention[0]
                col = dictionary[mention[1]]
                val = 1
                G.write(unicode(str(row) + " " + str(col) + " " + str(val) + "\n"))
    G.close()


def load_dict(file_path):
    elem_dict = {}
    elem_dict_pos = 0
    with io.open(file_path, 'r', encoding='utf8') as elem_dictionary:
        for row in elem_dictionary:
            elem_dict[row.replace("\n", "").split(",")[0]] = elem_dict_pos
            elem_dict_pos += 1
        elem_dictionary.close()
    return elem_dict


def load_mentions(file_path):
    elem_ment = []
    with io.open(file_path, 'r', encoding='utf8') as elem_mentions:
        for row in elem_mentions:
            elem_ment.append(row.replace("\n", "").split(":")[5])
        elem_mentions.close()
    return zip(xrange(len(elem_ment)), elem_ment)


def load_parsed_documents(file_path):
    doc_tokens = []
    with io.open(file_path, 'r', encoding='utf8') as document_tokens:
        for row in document_tokens:
            doc_tokens.append(row.replace("\n", "").split(":"))
        document_tokens.close()
    return doc_tokens


def get_em_context(corpus_mention_idx, doc_no, sent_no, corpus_ems):
    em_context = []
    for corpus_em in corpus_ems:
        corpus_em_no = int(corpus_em[0])
        corpus_doc_no = int(corpus_em[1])
        corpus_sent_no = int(corpus_em[2])
        if corpus_doc_no < doc_no:
            continue
        if corpus_doc_no > doc_no:
            break
        if corpus_doc_no == doc_no and corpus_sent_no == sent_no and corpus_mention_idx != corpus_em_no:
            em_context.append(corpus_em[3])
    return em_context


def get_rp_context(corpus_rp_idx, doc_no, sent_no, corpus_tokens, context_window):
    rp_context_sentence = []
    rp_position = 0
    context_len = 0
    for corpus_token in corpus_tokens:
        corpus_token_doc_no = int(corpus_token[0])
        corpus_sentence_no = int(corpus_token[1])
        corpus_token_no = int(corpus_token[3])
        if corpus_token_doc_no < doc_no:
            continue
        if corpus_token_doc_no > doc_no:
            break
        # sent no removed because there are no context for some relation phrases
        # if corpus_token_doc_no == doc_no and corpus_sentence_no == sent_no and corpus_token[9] == u"False" and corpus_token[-1] != u"EM":
        if corpus_token_doc_no == doc_no and corpus_token[9] == u"False" and corpus_token[-1] != u"EM":
            # TODO move checkers to text processing
            if corpus_token[-1] != u"RP" and not hasNumbers(corpus_token[5]) and len(corpus_token[5]) > 1:
                rp_context_sentence.append(corpus_token[5])
                context_len += 1
            if corpus_token[-1] == u"RP" and corpus_sentence_no == sent_no and corpus_token_no == corpus_rp_idx:
                rp_context_sentence.append(corpus_token[5])
                rp_position = context_len
                context_len += 1

    left_bound = rp_position - context_window
    right_bound = rp_position + context_window

    if left_bound >= 0:
        context = rp_context_sentence[left_bound:rp_position]
    else:
        context = rp_context_sentence[:rp_position]

    if right_bound <= context_len - 1:
        context += rp_context_sentence[rp_position + 1:right_bound]
    else:
        context += rp_context_sentence[rp_position + 1:]

    return context


def to_dict_from_set(words_set, filename):
    words_dict = {}
    with io.open(filename, 'a', encoding='utf8') as w_dict:
        for i, word in enumerate(words_set):
            words_dict[word] = i
            w_dict.write(unicode(word + "\n"))
        w_dict.close()
    return words_dict


def compute_tf_idf(tf, max_tf, idf, doc_num):
    tf_ = 0.5 + 0.5 * tf / max_tf
    tf_idf = tf_ * log(doc_num / (1.0 + idf))
    return tf_idf


def get_mention_if_idf(corpus_ems):
    ems_tf = Counter()
    ems_idf = {}
    for corpus_em in corpus_ems:
        doc_no = corpus_em[1]
        lemma = corpus_em[3]
        ems_tf[lemma] += 1
        if lemma not in ems_idf:
            ems_idf[lemma] = [doc_no]
        else:
            if doc_no not in ems_idf[lemma]:
                ems_idf[lemma].append(doc_no)

    return ems_tf, ems_idf


def construct_ment_ment_G(corpus_tokens, entity_dictionary, clusters_num):
    em_em_G = {}
    corpus_len = len(corpus_tokens)
    corpus_idx = xrange(corpus_len)
    doc_num = len(set([corpus_token[0] for corpus_token in corpus_tokens]))

    # (glob_doc_token_no, doc_no, sent_no, lemma)
    corpus_ems = [(i, corpus_tokens[i][0], corpus_tokens[i][1], corpus_tokens[i][5]) for i in corpus_idx if
                  corpus_tokens[i][-1] == u"EM"]

    doc_no = 0
    corpus_docs = []
    corpus_docs_ems = []
    for corpus_token in corpus_ems:
        corpus_doc_no = int(corpus_token[1])
        if doc_no == corpus_doc_no:
            corpus_docs_ems.append(corpus_token)
        if doc_no < corpus_doc_no:
            corpus_docs.append(corpus_docs_ems)
            doc_no += 1
            corpus_docs_ems = [corpus_token]

    corpus_docs.append(corpus_docs_ems)

    # (mention_no, (doc_token_no, doc_no, sent_no, lemma))
    ems = zip(xrange(len(corpus_ems)), corpus_ems)

    print "ENTITY MENTION CONTEXT: CORPUS LENGTH: " + str(corpus_len) + "ENTITY MENTIONS LENGTH: " + str(len(corpus_ems)) + " DOCS NUM: " + str(doc_num)

    mentions_context_gen_start_time = time.time()
    print "MENTION MENTION CONTEXT GENERATION START AT: " + str(mentions_context_gen_start_time)

    em_context = {}
    em_proceeded = 0
    for em in ems:
        em_no = em[0]
        em_corpus_idx = int(em[1][0])
        em_doc_no = int(em[1][1])
        em_sent_no = int(em[1][2])
        em_token = em[1][3]
        em_corpus_doc = corpus_docs[em_doc_no]
        em_tuple = (em_no, get_em_context(em_corpus_idx, em_doc_no, em_sent_no, em_corpus_doc))
        if em_token not in em_context:
            em_context[em_token] = [em_tuple]
        else:
            em_context[em_token].append(em_tuple)

        em_proceeded += 1
        if em_proceeded % 100 == 0:
            mention_processed_time = time.time()
            print str(em_proceeded) + " MENTIONS PROCESSED FOR: " + str(mention_processed_time - mentions_context_gen_start_time)

    key_list = em_context.keys()
    key_list = sorted(key_list)
    with io.open("entity_mention_context.txt", 'a', encoding='utf8') as context:
        for em_token in key_list:
            for em_tuple in em_context[em_token]:
                context.write(unicode(str(em_token) + "|" + str(em_tuple[0]) + ":" + ";".join(em_tuple[1]) + "\n"))
        context.close()

    m_m_G_gen_start_time = time.time()
    print "MENTION MENTION GRAPH GENERATION START AT: " + str(m_m_G_gen_start_time)

    ems_tf, ems_idf = get_mention_if_idf(corpus_ems)

    entity_dict_size = len(entity_dictionary.keys())  # column shape
    miss_em = []
    max_tf = max(ems_tf.iteritems(), key=itemgetter(1))[1]
    em_proceeded = 0
    for em_token in key_list:
        val = []
        row = []
        col = []
        ems_id_map = {}
        counter = 0  # row shape
        for em_tuple in em_context[em_token]:
            if len(em_tuple[1]) > 0:
                ems_id_map[counter] = em_tuple[0]
                for entity in em_tuple[1]:
                    row.append(counter)
                    entity_tf = ems_tf[entity]
                    entity_idf = len(ems_idf[entity])
                    col.append(entity_dictionary[entity])
                    val.append(compute_tf_idf(entity_tf, max_tf, entity_idf, doc_num))
                counter += 1

        k_adapt = min(clusters_num, counter - 1)
        if k_adapt > 0:
            if counter == 1:  # entity_mentions_size > 1 and only one not empty mention
                for em_tuple in em_context[em_token]:
                    miss_em.append(em_tuple[0])

            if counter > 1:  # need at least two mentions
                G_mention_context = coo_matrix((val, (row, col)), shape=(counter, entity_dict_size)).tocsr()
                # normalize each row to unit vector
                G_mention_context = normalize(G_mention_context, norm='l2', axis=1, copy=True)
                # KNN with Euclidean distance, possible to use Jaccard distance
                KNN_graph = kneighbors_graph(G_mention_context, k_adapt, mode='distance')
                ridx, cidx = KNN_graph.nonzero()
                for i in range(len(ridx)):
                    em_1 = ems_id_map[ridx[i]]
                    em_2 = ems_id_map[cidx[i]]
                    sim = exp(-KNN_graph[ridx[i], cidx[i]] ** 2 / 2)  # heat kernel with t=2
                    em_em_G[(em_1, em_2)] = sim
                    em_em_G[(em_2, em_1)] = sim
        else:
            miss_em.append(em_context[em_token][0][0])

        em_proceeded += 1
        if em_proceeded % 100 == 0:
            mention_processed_time = time.time()
            print str(em_proceeded) + " MENTIONS PROCESSED FOR: " + str(mention_processed_time - m_m_G_gen_start_time)

    with io.open("G_M.txt", 'a', encoding='utf8') as G_M:
        for em_token, value in em_em_G.items():
            G_M.write(unicode(str(em_token[0]) + " " + str(em_token[1]) + " " + str(value) + "\n"))
        G_M.close()

    with io.open("missed_G_M.txt", 'a', encoding='utf8') as missed_G_M:
        for missed in miss_em:
            missed_G_M.write(unicode(str(missed) + "\n"))
        missed_G_M.close()


def construct_relation_phrase_context(corpus_tokens, rp_dictionary, vocablulary, context_window):
    corpus_len = len(corpus_tokens)
    corpus_idx = xrange(corpus_len)
    doc_num = len(set([corpus_token[0] for corpus_token in corpus_tokens]))

    # (doc_token_no, doc_no, sent_no, lemma)
    corpus_rps = [(corpus_tokens[i][3], corpus_tokens[i][0], corpus_tokens[i][1], corpus_tokens[i][5]) for i in corpus_idx
                  if corpus_tokens[i][-1] == u"RP"]

    doc_no = 0
    corpus_docs = []
    corpus_docs_tokens = []
    for corpus_token in corpus_tokens:
        corpus_doc_no = int(corpus_token[0])
        if doc_no == corpus_doc_no:
            corpus_docs_tokens.append(corpus_token)
        if doc_no < corpus_doc_no:
            corpus_docs.append(corpus_docs_tokens)
            doc_no += 1
            corpus_docs_tokens = [corpus_token]

    corpus_docs.append(corpus_docs_tokens)

    # (rp_no, (doc_token_no, doc_no, sent_no, lemma))
    # rps = zip(xrange(len(corpus_rps)), corpus_rps)

    print "RELATION PHRASE CONTEXT: CORPUS LENGTH: " + str(corpus_len) + " RELATION PHRASES LENGTH: " + str(len(corpus_rps)) + " DOCS NUM: " + str(doc_num)

    feature_gen_start_time = time.time()
    print "FEATURE GENERATION START AT: " + str(feature_gen_start_time)

    rp_inner_words = {}
    unique_rp_words = set()
    for rp, rp_idx in rp_dictionary.items():
        rp_words = rp.split(" ")
        rp_inner_words[rp_idx] = rp_words
        for word in rp_words:
            unique_rp_words.add(word)

    with io.open("F_c.txt", 'a', encoding='utf8') as Fc:
        rp_words_dict = to_dict_from_set(unique_rp_words, "rp_words_dict.txt")
        for key, value in rp_inner_words.items():
            for word in value:
                Fc.write(unicode(str(key) + " " + str(rp_words_dict[word]) + " " + str(1) + "\n"))
        Fc.close()

    relation_phrase_features = {}
    unique_rp_context_words = set()
    rp_proceeded = 0
    for relation_phrase in corpus_rps:
        rp_corpus_idx = int(relation_phrase[0])
        rp_doc_no = int(relation_phrase[1])
        rp_sent_no = int(relation_phrase[2])
        rp_token = relation_phrase[3]
        rp_dict_idx = int(rp_dictionary[rp_token])
        rp_corpus_doc = corpus_docs[rp_doc_no]
        rp_tuple = (rp_token, get_rp_context(rp_corpus_idx, rp_doc_no, rp_sent_no, rp_corpus_doc, context_window))
        for word in rp_tuple[1]:
            unique_rp_context_words.add(word)

        if rp_dict_idx not in relation_phrase_features:
            relation_phrase_features[rp_dict_idx] = [rp_tuple]
        else:
            relation_phrase_features[rp_dict_idx].append(rp_tuple)

        rp_proceeded += 1
        if rp_proceeded % 100 == 0:
            features_processed_time = time.time()
            print str(rp_proceeded) + " FEATURES PROCESSED FOR: " + str(features_processed_time - feature_gen_start_time)

    # key_list = relation_phrase_features.keys()
    # key_list = sorted(key_list)
    with io.open("rp_features.txt", 'a', encoding='utf8') as rp_features:
        for rp_dict_idx, features in relation_phrase_features.items():
            for feature in features:
                rp_features.write(unicode(str(rp_dict_idx) + "|" + str(feature[0]) + ":" + ";".join(feature[1]) + "\n"))
        rp_features.close()

    with io.open("F_s.txt", 'a', encoding='utf8') as Fs, \
            io.open("missed_features.txt", 'a', encoding='utf8') as missed_features:
        rp_context_words_dict = to_dict_from_set(unique_rp_context_words, "rp_context_words_dict.txt")
        for rp_dict_idx, features in relation_phrase_features.items():
            unique_context_words = set()
            for feature in features:
                for word in feature[1]:
                    unique_context_words.add(word)

            if len(unique_context_words) > 0:
                for word in unique_context_words:
                    Fs.write(unicode(str(rp_dict_idx) + " " + str(rp_context_words_dict[word]) + " " + str(vocablulary[word]) + "\n"))
            else:
                missed_features.write(unicode(str(rp_dict_idx) + "\n"))

        Fs.close()
        missed_features.close()


if __name__ == "__main__":
    out_path = "/Users/Reist/PycharmProjects/prepate_data/processed_text/"

    text_corpus = []
    with io.open(out_path + "raw_documents.txt", 'r', encoding='utf8') as raw_documents:
        for line in raw_documents:
            text_corpus.append(line.replace("\n", ""))
    raw_documents.close()

    vocab = tf_idf_vocabulary(text_corpus)
    word2vec_model = word2vec_vocabulary(text_corpus)

    doc_tokens = load_parsed_documents(out_path + "parsed_documents.txt")

    entity_dict = load_dict(out_path + "entity_dictionary.txt")
    relation_dict = load_dict(out_path + "relation_dictionary.txt")

    entity_ment = load_mentions(out_path + "entitiy_mentions.txt")
    l_entity_rel_ment = load_mentions(out_path + "left_entity_relation.txt")
    r_entity_rel_ment = load_mentions(out_path + "right_entity_relation.txt")

    construct_biadjacency_G(entity_ment, entity_dict, "G_Q.txt")
    construct_biadjacency_G(l_entity_rel_ment, relation_dict, "G_L.txt")
    construct_biadjacency_G(r_entity_rel_ment, relation_dict, "G_R.txt")

    construct_ment_ment_G(doc_tokens, entity_dict, 3)
    construct_relation_phrase_context(doc_tokens, relation_dict, vocab, 5)
