import numpy as np
import re


class DataLoader(object):
    def __init__(self, filePath, encoding=None):
        self.filePath = filePath
        self.encoder = encoding

    def read_file(self):
        f = open(self.filePath, "r", encoding=self.encoder)
        s = f.read()
        return s

    # làm sạch data, tách dữ liệu thành 2 phần: label - text
    def get_data(self, labeled_data):
        X = []
        y = []
        # xử lý TH: label và text được phân cách khác nhau giữa các dòng
        labeled_data = re.sub(r'[\t\x0b\r\f+]', " ", labeled_data)
        data = np.array(labeled_data.split("\n"))
        for line in data:
            try:
                line = line.split(" ", 1)
                X.append(line[1])
                y.append(line[0])
            except:
                continue
        return X, y
