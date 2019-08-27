import pandas as pd


class DataLoader(object):
    ''' loading data file '''

    def __init__(self, filePath, encoding='latin-1'):
        self.filePath = filePath
        self.encoding = encoding

    def readFile(self):
        # nrows=k: limit num of row read
        file_content = pd.read_csv(self.filePath, names=['userId', 'movieId', 'rating'], skiprows=1, nrows=2000000,
                                   usecols=['userId', 'movieId', 'rating'])
        return file_content.as_matrix()
