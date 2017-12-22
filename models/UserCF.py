from sklearn.metrics.pairwise import cosine_similarity
from models.basemodel import basemodel
from scipy.sparse import csr_matrix
import numpy as np

class userCF(basemodel):
    """基于用户的协同过滤方法，由于点击数据大多都只有0,1，直接用与当前用户最相似
    的K个用户的评分加权预测对新项目的评分
    """
    def __init__(self, user_news_df, knn):
        basemodel.__init__(self, user_news_df)
        # 用cosine相似计算
        self.user_sim = cosine_similarity(self.ui_mat)
        self.knn = knn

    def train(self):
        # 计算所有用户的K近邻用户,
        self.sorted_sim = np.argsort(-self.user_sim, axis=1)

    def predict(self, user, item):
        # 找出K近邻用户集
        user_k = self.sorted_sim[user]
        user_topK = []
        k = 0
        for u in user_k:
            if k >= self.knn:
                break
            if self.ui_mat[u, item] > 0:
                user_topK.append(u)
                k += 1

        if len(user_topK) == 0 or self.user_sim[user_topK[0], user] < 1e-10:
            return 0

        prediction = 0
        sim_sum = 0
        for uk in user_topK:
            sim_sum += self.user_sim[user, uk]
            uki = self.ui_mat[uk, item]
            prediction += self.user_sim[user, uk] * uki
        if sim_sum < 1e-10:
            prediction = 0
        else:
            prediction = prediction / sim_sum
        return prediction
