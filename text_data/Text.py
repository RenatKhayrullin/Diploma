# coding=utf-8
import operator
import sys
import io
import os
import copy
import csv
import regex as re
from collections import Counter

from pymystem3 import Mystem
from stop_words import get_stop_words
from nltk.corpus import stopwords
from Token import Token
from Token import DocumentToken
from TextCleaner import TextCleaner
from nltk.tokenize import sent_tokenize

reload(sys)
sys.setdefaultencoding('utf-8')

language = "russian"
lang = "ru"


# TODO  need to correct punctuation sentence division : remove abbreviations
# TODO  need to correct markup
class TextTokens:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

        self.morph_analyzer = Mystem()

        stop_words = stopwords.words(language)
        stop_words += get_stop_words(lang)
        self.stws = set()
        for stw in stop_words:
            self.stws.add(stw.decode("utf8"))

        self.corpus = []
        self.corpus_entities = []
        self.corpus_relations = []

        self.raw_manual_markup = []
        self.raw_corpus_markup = []

        self.RP_CANDIDATE = ["PR", "V"]
        self.RP_PART = ["A", "ADV", "S", "PART", "ANUM", "NUM"]
        self.RP_PATTERN = r"PR{1}|[V](?:\s|V|PR){0,}"

    def check_rp_start(self, token): return token.pos_tag in self.RP_CANDIDATE

    def check_rp_part(self, token): return token.pos_tag in self.RP_PART

    def check_rp_pattern(self, token): return re.match(self.RP_PATTERN, token.pos_tag)[0] == token.pos_tag

    @staticmethod
    def create_analyzed_token(token_morph):
        token = token_morph["text"]
        lemma = None
        pos_tag = None
        if "analysis" in token_morph:
            word_analysis = token_morph["analysis"]
            if word_analysis:
                lemma = word_analysis[0]["lex"]
                grammar = str(word_analysis[0]["gr"]).replace("=", ",").split(",")
                pos_tag = grammar[0]
        return Token(token.strip(), str(lemma).strip(), str(pos_tag).strip())

    def tokenize_text(self, text):
        text_morph = self.morph_analyzer.analyze(text)
        # remove points from text
        text_morph = [token_morph for token_morph in text_morph
                      if token_morph["text"].strip() != "."
                      and token_morph["text"].strip() != "\n"
                      and len(token_morph["text"].strip()) > 0]
        return [self.create_analyzed_token(token_morph) for token_morph in text_morph]

    def load_markup(self, file_name):
        f_basename = os.path.splitext(file_name)[0]
        f_markup = f_basename + ".markup"
        path_to_markup = str(os.path.join(self.input_path, f_markup))
        if os.path.exists(path_to_markup) and os.path.isfile(path_to_markup):
            with io.open(path_to_markup, 'r', encoding='utf8') as f_data_markup:
                data_markup = []
                for line in f_data_markup:
                    row_data = line.replace("\n", "").decode("utf8").split(" ")
                    markup_tokens = unicode(" ".join([token for token in row_data[1:]]))
                    analyzed_tokens = self.tokenize_text(markup_tokens)
                    for token in analyzed_tokens:
                        token.token_type = row_data[0].upper().decode("utf8")
                    data_markup.append(analyzed_tokens)
                f_data_markup.close()
                return data_markup
        else:
            print("MARKUP FILE for " + str(file_name) + "DOES NOT EXIST")
            return []

    def processing_data_texts(self):
        doc_no = 0
        path_to_log = str(os.path.join(out_path, "LOG.txt"))
        with io.open(path_to_log, 'a', encoding='utf8') as f_log:
            for f_data in os.listdir(self.input_path):
                if f_data.endswith(".txt"):
                    path_to_text = str(os.path.join(self.input_path, f_data))
                    with io.open(path_to_text, 'r', encoding='utf8') as f_input:
                        self.raw_manual_markup.append(self.load_markup(f_data))
                        text = TextCleaner.clean_punctuation(f_input.read())
                        self.prepare_text(doc_no, text)
                        f_input.close()
                        f_log.write(unicode(str(doc_no) + ":" + str(f_data) + "\n"))
                        print "IN PROCESS : DOC " + str(doc_no) + ":" + str(f_data)
                    doc_no += 1
                    continue
                else:
                    continue
            f_log.close()

    def prepare_text(self, doc_no, doc_text):
        raw_markup = copy.deepcopy(self.raw_manual_markup[doc_no])
        sentences = sent_tokenize(doc_text, language=language)

        with io.open(str(os.path.join(out_path, "parsed_documents.txt")), 'a', encoding='utf8') as parsed_documents, \
                io.open(str(os.path.join(out_path, "raw_documents.txt")), 'a', encoding='utf8') as raw_documents, \
                io.open(str(os.path.join(out_path, "entitiy_mentions.txt")), 'a', encoding='utf8') as entitiy_mentions, \
                io.open(str(os.path.join(out_path, "relation_mentions.txt")), 'a', encoding='utf8') as relation_mentions, \
                io.open(str(os.path.join(out_path, "segmented_documents.txt")), 'a', encoding='utf8') as phr_seg:

            entity_markup_tokens = []  # for check markup

            corpus_tokens = []  # cleaned document
            document_entity_mentions = []  # entity mentions candidates for document
            document_relation_mentions = []  # entity relation candidates for document
            sentence_entities = []
            sentence_relations = []
            for i in xrange(len(sentences)):
                analyzed_sentence = TextCleaner.clean_sentence(sentences[i])
                # print "SENTENCE NO = "+str(i)+": "+sentences[i]
                sent_morph = self.tokenize_text(analyzed_sentence)
                sentence_partitions = []
                new_partition = []
                token_no = 0
                for j in xrange(len(sent_morph)):
                    token = sent_morph[j]
                    partition_no = len(sentence_partitions)
                    if token.word != ",":
                        token.is_stopword = True if token.lemma and token.lemma in self.stws else False
                        doc_token = DocumentToken(doc_no, i, partition_no, token_no, token)
                        new_partition.append(doc_token)
                        token_no += 1
                    else:
                        if len(new_partition) > 0:
                            sentence_partitions.append(new_partition)
                            new_partition = []
                if len(new_partition) > 0:
                    sentence_partitions.append(new_partition)

                ent_doc_token_type = "EM"
                for markup in raw_markup[:]:
                    continue_search = True
                    for partition in sentence_partitions:
                        partition_markups = []
                        if continue_search:
                            for doc_token in partition:
                                token = doc_token.token
                                if token in markup:
                                    partition_markups.append(doc_token)
                                else:
                                    if len(partition_markups) != len(markup):
                                        partition_markups = []

                                if len(partition_markups) > 0 and len(partition_markups) == len(markup):
                                    partition_tokens = [partition_markup.token for partition_markup in
                                                        partition_markups]

                                    diff_list = self.diff(markup, partition_tokens)

                                    if len(diff_list) == 0:
                                        # if raw_markup has multiple inclusions of the same markup
                                        # need to add markup only once
                                        at_least_one_token_exists = False
                                        for marked_token in partition_markups:
                                            for entity_markup in entity_markup_tokens:
                                                if marked_token in entity_markup:
                                                    at_least_one_token_exists = True
                                                    break

                                        # if partition_markups not in entity_markup_tokens:
                                        #     entity_markup_tokens.append(partition_markups)
                                        if not at_least_one_token_exists:
                                            entity_markup_tokens.append(partition_markups)

                                            token_type = markup[0].token_type
                                            raw_markup.remove(markup)

                                            entity_mention_phrase = None
                                            # merge markup tokens to phrase token
                                            for partition_markup in partition_markups:
                                                partition_markup.token.token_type = token_type
                                                # HACK-1
                                                # 1. some entity mentions marked as V (verb), for example "Снежеть" - V
                                                # 2. some entity mentions contain verbs, for example "газета Жить"
                                                partition_markup.doc_token_type = ent_doc_token_type
                                                if not entity_mention_phrase:
                                                    entity_mention_phrase = copy.deepcopy(partition_markup)
                                                else:
                                                    merged_entity_mention = Token.merge_tokens(entity_mention_phrase.token,
                                                                                               partition_markup.token,
                                                                                               token_type)
                                                    entity_mention_phrase.token = merged_entity_mention

                                            entity_mention_phrase.doc_token_type = ent_doc_token_type

                                            # HACK-2: because the same markup exists multiple times
                                            # in the one document markups
                                            if entity_mention_phrase not in document_entity_mentions:
                                                document_entity_mentions.append(entity_mention_phrase)
                                                sentence_entities.append(entity_mention_phrase)

                                        continue_search = False  # look for next markup
                                        break  # we found markup in partition, break loop for partition words
                                    else:
                                        partition_markups = []
                        else:
                            break

                rel_doc_token_type = "RP"
                for partition in sentence_partitions:
                    doc_relation_phrase = None
                    for doc_token in partition:
                        # from HACK-1
                        if doc_token.doc_token_type != "EM":
                            if self.check_rp_start(doc_token.token):
                                if not doc_relation_phrase:
                                    doc_relation_phrase = doc_token
                                    doc_relation_phrase.doc_token_type = rel_doc_token_type
                                else:
                                    relation_phrase_token = Token.merge_tokens(doc_relation_phrase.token, doc_token.token, None)
                                    if self.check_rp_pattern(relation_phrase_token):
                                        doc_relation_phrase.token = relation_phrase_token
                                    else:
                                        document_relation_mentions.append(doc_relation_phrase)
                                        sentence_relations.append(doc_relation_phrase)
                                        doc_relation_phrase = None
                            else:
                                if doc_relation_phrase:
                                    document_relation_mentions.append(doc_relation_phrase)
                                    sentence_relations.append(doc_relation_phrase)
                                    doc_relation_phrase = None
                        else:
                            if doc_relation_phrase:
                                document_relation_mentions.append(doc_relation_phrase)
                                sentence_relations.append(doc_relation_phrase)
                                doc_relation_phrase = None

                    if doc_relation_phrase:
                        document_relation_mentions.append(doc_relation_phrase)
                        sentence_relations.append(doc_relation_phrase)

                def save_entity_relations(entity, closest_relation, filename):
                    with io.open(str(os.path.join(out_path, filename)), 'a', encoding='utf8') as entity_relation:
                        relation_string = closest_relation.token.lemma if closest_relation else str(None)
                        entity_relation.write(unicode(str(entity.doc_no) + ":" + str(entity.sentence_no) + ":" +
                                                      str(entity.segment_no) + ":" + str(entity.token_no) + ":" +
                                                      entity.token.lemma + ":" + relation_string + "\n"))
                        entity_relation.close()

                for doc_entity in sentence_entities:
                    closest_left_distance = sys.maxint
                    closest_right_distance = -sys.maxint - 1
                    closest_left_relation = None
                    closest_right_relation = None
                    doc_entity_token_no = doc_entity.token_no
                    for doc_relation in sentence_relations:
                        rel_entity_token_no = doc_relation.token_no
                        distance = doc_entity_token_no - rel_entity_token_no
                        if 0 < distance < closest_left_distance:
                            closest_left_distance = distance
                            closest_left_relation = doc_relation
                        if 0 > distance > closest_right_distance:
                            closest_right_distance = distance
                            closest_right_relation = doc_relation

                    save_entity_relations(doc_entity, closest_left_relation, "left_entity_relation.txt")
                    save_entity_relations(doc_entity, closest_right_relation, "right_entity_relation.txt")

                def replace_tokens(sentence_partitions, sentence_entities):
                    new_sentence_partitions = []
                    for partition_idx in xrange(len(sentence_partitions)):
                        partition = sentence_partitions[partition_idx]
                        for entity in sentence_entities[:]:
                            entity_token_no = entity.token_no
                            for doc_token_idx in xrange(len(partition[:])):
                                doc_token_no = partition[doc_token_idx].token_no
                                if entity_token_no == doc_token_no:
                                    phrase_len = len(entity.token.lemma.split(" "))
                                    new_partition = partition[:doc_token_idx]
                                    remain_partition = partition[doc_token_idx:]
                                    for idx in xrange(phrase_len):
                                        if len(remain_partition) > 0:
                                            remain_partition.pop(0)
                                    new_partition += [entity]
                                    new_partition += remain_partition
                                    partition = new_partition
                                    sentence_entities.remove(entity)
                                    break
                        new_sentence_partitions.append(partition)
                    return new_sentence_partitions

                sentence_partitions = replace_tokens(sentence_partitions, sentence_entities)
                sentence_partitions = replace_tokens(sentence_partitions, sentence_relations)

                sentence_entities = []
                sentence_relations = []

                for k in xrange(len(sentence_partitions)):
                    segments = []
                    for document_token in sentence_partitions[k]:
                        parsed_documents.write(document_token.get_str())
                        token = document_token.token
                        if not token.is_stopword and len(token.lemma) > 0:
                            segments.append(token.lemma)
                    corpus_tokens += segments
                    phr_seg.write(unicode(str(" ".join(segments)) + "\n"))

            for token in document_relation_mentions:
                relation_mentions.write(token.get_str())
            for token in document_entity_mentions:
                entitiy_mentions.write(token.get_str())

            self.raw_corpus_markup.append(entity_markup_tokens)
            self.corpus_entities.append(document_entity_mentions)
            self.corpus_relations.append(document_relation_mentions)
            raw_documents.write(unicode(" ".join(corpus_tokens) + "\n"))

        relation_mentions.close()
        entitiy_mentions.close()
        raw_documents.close()
        phr_seg.close()
        parsed_documents.close()

    @staticmethod
    def diff(li1, li2):
        return list(set(li1) - set(li2))

    def save_dictionary(self):
        def create_dictionary(corpus_entities, filename):
            entity_dictionary = Counter()
            for doc_entities in corpus_entities:
                for doc_entity in doc_entities:
                    entity_dictionary[doc_entity.token.lemma] += 1
            with io.open(str(os.path.join(out_path, filename)), 'wb') as dictionary:
                writer = csv.writer(dictionary)
                sorted_dict = sorted(entity_dictionary.items(), key=operator.itemgetter(1))
                for key, value in sorted_dict:
                    writer.writerow([key, value])
            dictionary.close()
        create_dictionary(self.corpus_entities, "entity_dictionary.txt")
        create_dictionary(self.corpus_relations, "relation_dictionary.txt")

    def check_markup(self):
        docs_mp_len = len(self.raw_corpus_markup)
        for i in xrange(docs_mp_len):
            # markup for i-th document
            d_markup = copy.deepcopy(self.raw_corpus_markup[i])
            r_markup = copy.deepcopy(self.raw_manual_markup[i])
            # markup rows for i-th document
            for d_mp in d_markup[:]:
                for r_mp in r_markup[:]:
                    d_mp_tokens = [doc_token.token for doc_token in d_mp]
                    markup_diff = self.diff(d_mp_tokens, r_mp)
                    if len(markup_diff) == 0:
                        r_markup.remove(r_mp)
                        d_markup.remove(d_mp)
                        break

            with io.open(str(os.path.join(out_path, "raw_markup_errors.txt")), 'a', encoding='utf8') as r_markup_errors:
                if len(r_markup) > 0:
                    for r_mp in r_markup:
                        markup_words = [token.word for token in r_mp]
                        r_markup_errors.write(unicode(str(i) + " UNPAIRED MARKUP : " + " ".join(markup_words) + "\n"))
            r_markup_errors.close()

            with io.open(str(os.path.join(out_path, "data_markup_errors.txt")), 'a',
                         encoding='utf8') as d_markup_errors:
                if len(d_markup) > 0:
                    for d_mp in d_markup:
                        d_markup_words = [doc_token.token.word for doc_token in d_mp]
                        d_markup_errors.write(unicode(str(i) + " UNPAIRED MARKUP : " + " ".join(d_markup_words) + "\n"))
            d_markup_errors.close()


if __name__ == "__main__":
    input_path = "/Users/Reist/PycharmProjects/prepate_data/data/"
    out_path = "/Users/Reist/PycharmProjects/prepate_data/processed_text"
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    prepare_data = TextTokens(input_path, out_path)
    prepare_data.processing_data_texts()
    prepare_data.save_dictionary()
    prepare_data.check_markup()
    # pickle.dump(prepare_data.data, open("processed_text/data.pickle", "w"))  # serialize the data
