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

    def predict(self, user, item):
        # 找出K近邻用户集
        user_k = self.user_sim[user, :].argsort()[::-1]
        user_k = user_k[1: self.knn + 1]
        prediction = 0
        sim_sum = 0
        for uk in user_k:
            sim_sum += self.user_sim[user, uk]
            uki = self.ui_mat[uk, item]
            prediction += self.user_sim[user, uk] * uki
        prediction = prediction / sim_sum
        return prediction

    def predict_topK(self, user, K):
        # 找出K近邻用户集
        user_k = self.user_sim[user, :].argsort()[::-1]
        user_k = user_k[1: self.knn + 1]
        # user_rating = self.ui_mat.getrow(user)
        user_rating = self.ui_mat[user, :]

        rec_list = dict()
        for item in range(self.ITEM_NUM):
            # 对未评分的项目预测分数
            if user_rating[item] == 0:
                prediction = 0
                sim_sum = 0.0
                for uk in user_k:
                    sim_sum += self.user_sim[user, uk]
                    uki = self.ui_mat[uk, item]
                    prediction += self.user_sim[user, uk] * uki
                if sim_sum > 1e-8:
                    prediction = prediction / sim_sum
                else:
                    prediction = 0
                rec_list[item] = prediction
        # 取topK个项目生成推荐列表
        rec_topK = sorted(rec_list.items(), key=lambda e: e[1], reverse=True)
        return [rec_topK[i][0] for i in range(K)]
