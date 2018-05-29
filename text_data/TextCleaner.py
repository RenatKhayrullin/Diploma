import regex as re


class TextCleaner:
    def __init__(self):
        pass

    @staticmethod
    def clean_text(text):
        clear_text_regexp = re.compile(r'(?u)\w+|[,.!?]')
        text_ = " ".join(clear_text_regexp.findall(text)).replace(" .", ".").replace(" ,", ",")
        text_ = re.sub("[,]+", ",", text_)
        text_ = re.sub("[.]+", ".", text_)
        text_ = re.sub("\s+", " ", text_)
        return text_

    @staticmethod
    def clean_sentence(text):
        clear_sent_regexp = re.compile(r'(?u)\w+|[,.]')
        text_ = " ".join(clear_sent_regexp.findall(text)).replace(" .", ".").replace(" ,", ",")
        text_ = re.sub("[,]+", ",", text_)
        text_ = re.sub("[.]+", ".", text_)
        text_ = re.sub("\s+", " ", text_)
        return text_

    @staticmethod
    def clean_punctuation(text):
        text_ = text.replace("\t", " ")
        text_ = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*#,]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', "", text_)
        text_ = re.sub('\[({})\];:/\\]', ",", text_)
        line_breaks = [pos for pos, char in enumerate(text_) if char == "\n"]
        list_char = list(text_)
        for line_break in line_breaks:
            if list_char[line_break - 1] != "." \
                    and list_char[line_break - 1] != "\n" \
                    and list_char[line_break - 1] != " ":
                list_char[line_break] = "."
        text_ = "".join(list_char)
        text_ = TextCleaner.clean_text(text_)
        return text_