class Token:
    def __init__(self, word, lemma, pos_tag):
        self.word = word
        self.lemma = lemma if lemma != "None" else word.lower()
        self.pos_tag = pos_tag
        self.token_type = str(None)
        self.consecutive_cap = unicode(word)[0].isupper()
        self.is_stopword = False

    @staticmethod
    def merge_tokens(token1, token2, token_type):
        word = token1.word + " " + token2.word
        lemma = token1.lemma + " " + token2.lemma
        pos_tag = token1.pos_tag + " " + token2.pos_tag
        merged_token = Token(word, lemma, pos_tag)
        merged_token.consecutive_cap = token2.consecutive_cap and token1.consecutive_cap
        if token_type:
            merged_token.token_type = token_type

        return merged_token

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.word == other.word

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.word)

    def is_lemmas_equal(self, token):
        return self.lemma == token.lemma

    def get_str(self):
        return unicode(
            self.word + ":" +
            self.lemma + ":" +
            self.pos_tag + ":" +
            self.token_type + ":" +
            str(self.consecutive_cap) + ":" +
            str(self.is_stopword))


class DocumentToken:
    def __init__(self, doc_no, sentence_no, segment_no, token_no, token):
        self.doc_no = doc_no
        self.sentence_no = sentence_no
        self.segment_no = segment_no
        self.token_no = token_no
        self.token = token
        self.doc_token_type = str(None)

    def get_str(self):
        return unicode(
            str(self.doc_no) + ":" +
            str(self.sentence_no) + ":" +
            str(self.segment_no) + ":" +
            str(self.token_no) + ":" +
            self.token.get_str() + ":" +
            self.doc_token_type + "\n")

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.doc_no == other.doc_no and \
                   self.sentence_no == other.sentence_no and \
                   self.segment_no == other.segment_no and \
                   self.token_no == other.token_no

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(str(self.doc_no) + ":" +
                    str(self.sentence_no) + ":" +
                    str(self.segment_no) + ":" +
                    str(self.token_no))
