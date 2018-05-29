# coding=utf-8
import sys
import io
import math
import copy
import csv
import pickle
import regex as re
from collections import Counter
from text_data import Token

reload(sys)
sys.setdefaultencoding('utf-8')

out_path = "processed_text"


class PhraseExtractor:
    def __init__(self, data_path, frequent_patterns_path, significance):
        self.docs = pickle.load(open(data_path, 'r'))
        self.full_docs = []
        self.pos = []
        self.full_pos = []

        self.significance = significance
        self.vocabulary_size = 0

        self.frequent_patterns = Counter()
        with io.open(frequent_patterns_path, 'r', encoding='utf8') as word_freqs:
            reader = csv.reader(word_freqs)
            for row in reader:
                self.frequent_patterns[row[0].decode("utf8")] = int(row[1])
        word_freqs.close()

        for doc in self.docs:
            for sent in doc:
                for part in sent:
                    for word in part:
                        if not word.is_stopword:
                            self.vocabulary_size += 1

        print("VOCAB_SIZE: " + str(self.vocabulary_size))

    @staticmethod
    def phrase_tuple(phrase1, phrase2):
        return tuple(j for i in (phrase1, phrase2) for j in (i if isinstance(i, tuple) else (i,)))

    @staticmethod
    def merge_tokens(token1, token2):
        token_num = str(token1.token_num)+" "+str(token2.token_num)
        token = str(token1.token)+" "+str(token2.token)
        norm_token = str(token1.norm_token)+" "+str(token2.norm_token)
        pos_tag = str(token1.pos_tag)+" "+str(token2.pos_tag)
        pymorphy_pos_tag = str(token1.pymorphy_pos_tag) + " " + str(token2.pymorphy_pos_tag)
        merged_token = Token(token_num, token, norm_token, pos_tag, pymorphy_pos_tag)
        merged_token.start_from_cap = token2.start_from_cap and token1.start_from_cap
        return merged_token

    def significance_score(self, phrase1, phrase2):
        combined_phrase = phrase1 + " " + phrase2
        combined_occurence = self.frequent_patterns[combined_phrase]
        independent_occurence = self.frequent_patterns[phrase1] * self.frequent_patterns[phrase2]
        if combined_occurence == 0:
            return float("-inf")
        independent_prob = float(independent_occurence) / self.vocabulary_size
        numerator = float(combined_occurence) - independent_prob
        denominator = math.sqrt(combined_occurence)
        return numerator / denominator

    @staticmethod
    def check_entity(pattern):
        # r1 = re.match(r"[V]?[\sA]{0,}(?:\s|S)+", pattern)
        # r2 = re.match(r"(?:\s|None)+", pattern)
        r1 = re.match(r"[\sA]{0,}(?:\s|S)+", pattern)
        return True if r1 and r1[0] == pattern else False  # or (True if r2 and r2[0] == pattern else False)

    def document_partition(self, segment):
        significance_mapping = {}
        segment_partition = copy.copy(segment)
        while True:
            best_significance = float("-inf")
            for index in xrange(len(segment_partition[:-1])):
                phrase_1_str = segment_partition[index].norm_token.decode("utf8")
                phrase_2_str = segment_partition[index+1].norm_token.decode("utf8")
                phrase = phrase_1_str + " " + phrase_2_str
                merged_token = self.merge_tokens(segment_partition[index], segment_partition[index+1])

                if phrase not in significance_mapping:
                    sig_score = self.significance_score(phrase_1_str, phrase_2_str)
                    if self.check_entity(merged_token.pos_tag):
                        significance_mapping[phrase] = sig_score

                if phrase in significance_mapping and best_significance < significance_mapping[phrase]:
                    best_significance = significance_mapping[phrase]

                if phrase in significance_mapping and significance_mapping[phrase] > self.significance:
                    segment_partition[index] = merged_token
                    segment_partition.pop(index+1)
                    break

            if best_significance < self.significance:
                break

        # fix phrases
        return segment_partition

    def extract_entity_phrases(self):
        with io.open("processed_text/documents_partition.txt", 'a', encoding='utf8') as doc_part:
            for doc_no in xrange(len(self.docs)):  # iterate over documents
                print("Process document : " + str(doc_no))
                for sent_no in xrange(len(self.docs[doc_no])):  # iterate over sentences
                    for segment_no in xrange(len(self.docs[doc_no][sent_no])):  # iterate over segments
                        segment = self.docs[doc_no][sent_no][segment_no]
                        segment = [w for w in segment if not w.is_stopword]
                        if segment:
                            segment_partition = self.document_partition(segment)
                            if segment_partition:
                                segment_string = [w.token for w in segment_partition]
                                w_str = unicode(str(doc_no) + ":" + str(sent_no) + ":" + str(segment_no) + ":" + str(
                                    "|".join(segment_string)) + "\n")
                                doc_part.write(w_str)


if __name__ == "__main__":
    freq_pattens_path = "processed_text/freq_patterns_dict.csv"
    docs_path = "processed_text/data.pickle"
    PHR_EXT = PhraseExtractor(docs_path, freq_pattens_path, 1)
    PHR_EXT.extract_entity_phrases()
