import sys

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline

sys.path.insert(1, '../')
from processing import FeatureTransformer

class NaiveBayesModel(object):

    def __init__(self):
        self.clf_training = self._init_pipeline()

    @staticmethod
    def _init_pipeline():
        pipeline = Pipeline([
            ('transformer', FeatureTransformer()),
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', BernoulliNB())
        ])

        return pipeline
        # build vocab
        # X = vectorizer.fit_transform(x_train)
        # print(vectorizer.vocabulary_)

        # Fit model
        # train_model.fit(X, y_train)

        # save model
        # filename = "MultinomialNB"
        # model.save(train_model, filename)

        # return train_model
