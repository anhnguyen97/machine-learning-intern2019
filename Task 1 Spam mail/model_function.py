import pickle


class Model():

    def __init__(self):
        self.num = 2
        # self.save = self.save()
        # self.load = self.load()

    def save(model, filename):
        # path = "./models/".join(filename)
        # print("done")
        pickle.dump(model, open(filename, 'wb'))

    # def load(filename):
    #     # path = "./models/".join(filename)
    #     return pickle.load(open(filename, 'rb'))
