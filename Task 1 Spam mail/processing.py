from pyvi import ViTokenizer, ViUtils
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.num = None
        # self.tokenizer = ViTokenizer()
        # self.remove_accents = ViUtils()

    def fit(self, *_):
        return self

    def transform(self, dataset, y=None, **fit_params):
        process_data = []
        # load stop-words
        stop = set(line.strip() for line in open("vietnamese-stopwords.txt"))

        for line in dataset:
            # loai bo stopword
            text = ' '.join(str(word).lower() for word in line.split(" ") if str(word).lower() not in stop)

            # tokenizer, loai bo dau
            text = ViUtils.remove_accents(ViTokenizer.tokenize(text))

            process_data.append(text)

        # process_data = dataset.apply(lambda text: ViTokenizer.tokenize(text))

        print(process_data)
        return process_data
