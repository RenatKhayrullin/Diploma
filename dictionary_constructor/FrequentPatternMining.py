# coding=utf-8
__author__ = 'ahmed'
# source url: https://github.com/shanzhenren/ClusType

import sys
from collections import Counter
from BitVector import BitVector

reload(sys)
sys.setdefaultencoding('utf-8')


class FrequentPatternMining:
    def __init__(self, phrase_segments, max_pattern_size, minsup):
        self.phrase_segments = [None] * len(phrase_segments)  # phrase segments
        for i in xrange(len(phrase_segments)):
            self.phrase_segments[i] = phrase_segments[i].strip().split()  # tokenize each segment

        self.minsup = minsup
        self.max_pattern = max_pattern_size

        # for each pattern size create a frequent pattern counter:
        # Counter: { "obj_1":count_1, ... , "obj_n":count_n }
        self.frequent_patterns_counter = [Counter() for i in xrange(max_pattern_size)]  # note thar xrange returns 0 ... (n-1)

        self.SegmentApriori = [None] * len(self.phrase_segments)  # list of segments bit-representations

        for i in xrange(len(self.phrase_segments)):  # for each phrase segment
            self.SegmentApriori[i] = BitVector(size=len(self.phrase_segments[i]))  # create bir_vector of len of segment

    def mine(self, segment, pattern_size, global_patterns_counter, segment_apriori):
        segment_size = len(segment)
        continue_mining = False

        for i in xrange(segment_size + 1 - pattern_size):  # iterate over words in segment
            if segment_apriori[i] == 0:
                # HACK: for checking for end of array
                # HACK description:
                #           if first boolean clause is true the next boolean clause is not checked
                #           then there are no out of bound exception
                if i == segment_size - 1 or segment_apriori[i + 1] == 0:
                    cand = " ".join([segment[i + j] for j in xrange(pattern_size)])  # create candidate to a phrase of length = pattern_size
                    continue_mining = True

                    if not cand in self.frequent_patterns_counter[pattern_size - 1]:  # if word not in frequent pattern counter
                        curr_count = global_patterns_counter[cand] + 1  # update word count in global counter

                        if curr_count >= self.minsup:  # if word count satisfies to minsup condition
                            self.frequent_patterns_counter[pattern_size - 1][cand] = curr_count  # add word to frequent pattern counter
                            del global_patterns_counter[cand]  # remove word from global counter
                        else:  # if word doesn't satisfies to minsup condition
                            global_patterns_counter[cand] = curr_count  # update word count in global counter
                    else:  # if word in frequent pattern counter
                        self.frequent_patterns_counter[pattern_size - 1][cand] += 1  # update word frequency
                else:
                    segment_apriori[i] = 1  # when comes to the end of segment, mark last word as 1

        return continue_mining

    def mine_fixed_pattern(self, last_phrase_segment, pattern_size):
        index = 0  # index over phrase segments
        global_patterns_counter = Counter()  # create global words counter

        while index <= last_phrase_segment:
            segment = self.phrase_segments[index]  # get i-th phrase segment
            segment_apriori = self.SegmentApriori[index]  # get i-th phrase segment bit-representation the same shape as segment

            # after the first iteration with pattern_size = 1 the self.SegmentApriori[segment_size -1] will be 1
            # after the second iteration with pattern_size = 2 the self.SegmentApriori[segment_size -2] will be 1
            # ...etc
            continue_mining = self.mine(segment, pattern_size, global_patterns_counter, segment_apriori)

            # check length of segment
            # the all words except the first is 1 or segment size less that pattern_size
            if not continue_mining or len(segment) <= pattern_size:
                # remove segment from further analysis
                # swap current segment with the last segment
                self.phrase_segments[index], self.phrase_segments[last_phrase_segment] = self.phrase_segments[last_phrase_segment], self.phrase_segments[index]
                self.SegmentApriori[index], self.SegmentApriori[last_phrase_segment] = self.SegmentApriori[last_phrase_segment], self.SegmentApriori[index]
                last_phrase_segment -= 1
            else:
                index += 1

        del global_patterns_counter

        pattern_size += 1

        print "Ending Mining of Patterns of Size: " + str(pattern_size - 1)
        print "Documents Remaining: " + str(last_phrase_segment)

        return last_phrase_segment

    def mine_patterns(self):
        pattern_size = 1
        last_phrase_segment = len(self.phrase_segments) - 1
        print "Mining Contiguous Patterns"

        while last_phrase_segment >= 0:  # iterate over documents
            # for each document mine frequent patterns of length = pattern_size
            last_phrase_segment = self.mine_fixed_pattern(last_phrase_segment, pattern_size)
            pattern_size += 1
            if pattern_size > self.max_pattern:
                break

        return self.frequent_patterns_counter


if __name__ == "__main__":
    '''
    input='Intermediate/segmented_documents.txt'
    maxLength='5'
    minSup='2'
    '''
    #path = sys.argv[1]
    max_pattern_size = 5  # int(sys.argv[2])
    minsup = 2  # int(sys.argv[3])
    phrase_segments = []

    with open("processed_text/segmented_documents.txt", 'r') as f:
        for row in f:
            text = row.strip()
            if len(text) > 0:
                phrase_segments.append(text)

    FPM = FrequentPatternMining(phrase_segments, max_pattern_size, minsup)
    FrequentPatterns = FPM.mine_patterns()

    import csv
    with open('processed_text/freq_patterns_dict.csv', 'wb') as csv_file:
        writer = csv.writer(csv_file)
        for i in xrange(max_pattern_size):
            for key, value in FrequentPatterns[i].items():
                writer.writerow([key, value])
