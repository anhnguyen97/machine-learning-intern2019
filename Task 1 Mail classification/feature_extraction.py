import pickle

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class FeatureExtraction(object):

    def __init__(self, X_train=None, X_test=None):
        self.X_train = X_train
        self.X_test = X_test

    def save_vect(self, filePath, vect):
        pickle.dump(vect, open(filePath, 'wb'))

    def load_vect(self, filePath):
        return pickle.load(open(filePath, "rb"))

    def count_vect(self):
        # create an Count vectorizer object
        count_vect = CountVectorizer(analyzer="word")
        # build vocab of all tokens in the raw documents
        count_vect.fit(self.X_train)

        # transform the training and validation data using count vectorizer object
        X_train_count = count_vect.transform(self.X_train)
        # X_test_count = count_vect.transform(self.X_test)

        self.save_vect('./vect/count_vect.pkl', count_vect)

        return X_train_count
        # return X_train_count, X_test_count

    def word_tfidf_vect(self):
        # word level - we choose max number of words equal to 30000 except all words (100k+ words)
        tfidf_vect = TfidfVectorizer(analyzer='word', max_features=30000)
        tfidf_vect.fit(self.X_train)  # learn vocabulary and idf from training set

        X_train_word_tfidf = tfidf_vect.transform(self.X_train)
        # X_test_word_tfidf = tfidf_vect.transform(self.X_test)
        # return X_train_word_tfidf, X_test_word_tfidf

        self.save_vect('./vect/word_tfidf_vect.pkl', tfidf_vect)

        return X_train_word_tfidf

    def ngram_tfidf_vect(self):
        # ngram level - we choose max number of words equal to 30000 except all words (100k+ words)
        tfidf_vect_ngram = TfidfVectorizer(analyzer='word', max_features=30000, ngram_range=(2, 3))
        tfidf_vect_ngram.fit(self.X_train)

        X_train_ngram_tfidf = tfidf_vect_ngram.transform(self.X_train)
        X_test_ngram_tfidf = tfidf_vect_ngram.transform(self.X_test)

        return X_train_ngram_tfidf, X_test_ngram_tfidf
