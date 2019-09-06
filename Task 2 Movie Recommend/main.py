from CF import *
from DataLoader import *
import numpy as np
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    filePath = "./data/ml-20m/ratings.csv"

    print("Loading data:")
    dataLoader = DataLoader(filePath)
    Y_data = dataLoader.readFile()
    print("Done!\n--------------------")

    y = np.zeros(Y_data.shape[0])
    rate_train, rate_test, y_train, y_test = train_test_split(Y_data, y, test_size=0.2)

    print("User-user Collaborative Filtering: ")
    cf = CF(rate_train, 5)
    cf.fit()
    # cf.print_recommendation()
    print("-------------------------")

    print("\nTESTING: \nComputing RMSE: ")
    cf.RMSE(rate_test)
