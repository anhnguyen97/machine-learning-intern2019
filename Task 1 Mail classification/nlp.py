import gensim
from DataLoader import *
from pyvi import ViTokenizer

# Processing problem about text: remove stopword, tokenizer, remove special character
class NLP(object):

    def __init__(self, text=None):
        self.text = None
        self.__set_stopwords()

    def __set_stopwords(self):
        data_loader = DataLoader('./vietnamese-stopwords.txt')
        content = data_loader.read_file()
        stop_words_set = set(line.strip() for line in content.split("\n"))
        stop_words_set = set(stop_word.strip().replace(" ", "_") for stop_word in stop_words_set)
        self.stop_words = stop_words_set

    def segmentation(self):
        return ViTokenizer.tokenize(self.text)

    def remove_special_character(self):
        return gensim.utils.simple_preprocess(self.text)

    def preprocessing(self, X_train):
        stop_words = self.stop_words
        preprocessed_x_train = []
        for line in X_train:
            self.text = line
            # remove special character, return list word
            after_remove_special = self.remove_special_character()
            self.text = " ".join(after_remove_special)

            # tokenizer: tách từ
            after_tokenize = self.segmentation()

            # remove stopwords
            word_set = set(word for word in after_tokenize.split(" ") if word not in stop_words)
            after_remove_stop = " ".join(word_set)

            preprocessed_x_train.append(after_remove_stop)

        return preprocessed_x_train

