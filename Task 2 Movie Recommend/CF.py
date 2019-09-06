import numpy as np
from DataLoader import *
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity


class CF(object):

    def __init__(self, Y_data, k, dist_func=cosine_similarity):
        self.Y_data = Y_data  # original data
        self.k = k  # number of neighbor points
        self.dist_func = dist_func  # function for similarity between users
        self.Ybar_data = None  # save normalized data

        # number of users and items
        self.n_users = int(np.max(self.Y_data[:, 0])) + 1
        self.n_items = int(np.max(self.Y_data[:, 1])) + 1

    def normalize_Y(self):
        users = self.Y_data[:, 0]
        self.Ybar_data = self.Y_data.copy()
        self.mu = np.zeros((self.n_users,))

        for n in range(self.n_users):
            # line in file have user-n rating
            index_rating = np.where(users == n)[0].astype(np.int32)

            # list rating that user-n rated
            ratings = self.Y_data[index_rating, 2]

            # mean of user-n ratings
            if np.size(ratings) > 0 and not np.isnan(ratings).any():  # to avoid empty array and nan value
                mean_ratings = np.mean(ratings)
            else:
                mean_ratings = 0

            self.mu[n] = mean_ratings

            # normalize
            self.Ybar_data[index_rating, 2] = ratings - self.mu[n]

        # print(self.Ybar_data)

        ################################################
        # form the rating matrix as a sparse matrix. store
        # nonzeros only, and, of their locations.
        self.Ybar = sparse.coo_matrix((self.Ybar_data[:, 2], (self.Ybar_data[:, 1], self.Ybar_data[:, 0])),
                                      (self.n_items, self.n_users))
        self.Ybar = self.Ybar.tocsr()

    def similarity(self):
        """
        Computing the similarity btw users
        :return: none
        """
        self.S = self.dist_func(self.Ybar.T, self.Ybar.T)

    def fit(self):
        """
        Fit model when has a new data
        :return: none
        """
        self.normalize_Y()
        self.similarity()

    def pred(self, u, i, normalized=1):
        """
        predict the rating of user u for item i (normalized)
        """
        # Step 1: find all users who rated i
        ids = np.where(self.Y_data[:, 1] == i)[0].astype(np.int32)
        # Step 2:
        users_rated_i = (self.Y_data[ids, 0]).astype(np.int32)
        # Step 3: find similarity btw the current user and others
        # who already rated i
        sim = self.S[u, users_rated_i]
        # Step 4: find the k most similarity users, get index of k most similarity users
        a = np.argsort(sim)[-self.k:]
        # and the corresponding similarity levels
        nearest_s = sim[a]
        # How did each of 'near' users rated item i
        r = self.Ybar[i, users_rated_i[a]]
        if normalized:
            # add a small number, for instance, 1e-8, to avoid dividing by 0
            return (r * nearest_s)[0] / (np.abs(nearest_s).sum() + 1e-8)

        return (r * nearest_s)[0] / (np.abs(nearest_s).sum() + 1e-8) + self.mu[u]

    def recommend(self, userId):
        """
        :param userId: user who need to be recommended
        :return: list item that user-u can interested in
        """
        ids = np.where(self.Y_data[:, 0] == userId)[0]  # index of line userId rated
        items_rated_by_u = self.Y_data[ids, 1].tolist()
        # print(items_rated_by_u)
        recommended_items = []
        for i in range(self.n_items):
            if i not in items_rated_by_u:
                rating = self.pred(userId, i)
                if rating > 0:
                    recommended_items.append(i)

        return recommended_items

    def print_recommendation(self):
        """
        print all items which should be recommended for each user
        """
        for u in range(self.n_users):
            recommended_items = self.recommend(u)
            print('Recommend item(s):', recommended_items, 'to user', u)

    def RMSE(self, rate_test):
        n_tests = rate_test.shape[0]
        SE = 0  # squared error
        userlist = self.Y_data[:, 0].tolist()

        for n in range(n_tests):
            if rate_test[n, 0] in userlist:
                pred = self.pred(int(rate_test[n, 0]), int(rate_test[n, 1]), normalized=0)
                SE += (pred - rate_test[n, 2]) ** 2

        RMSE = np.sqrt(SE / n_tests)
        print('User-user CF, RMSE={}'.format(RMSE))
        return RMSE
