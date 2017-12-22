from sklearn.metrics.pairwise import cosine_similarity
from models.basemodel import basemodel
from scipy.sparse import csr_matrix
import numpy as np

class itemCB(basemodel):
    """基于项目内容的K近邻方法，计算相似项目使用新闻内容的tfidf值，而不采用评分
    计算该用户对的最近似K个项目的评分加权预测对该项目的评分
    """
    def __init__(self, user_news_df, knn):
        basemodel.__init__(self, user_news_df)
        # 用cosine相似计算
        self.knn = knn

    def train(self):
        self.item_sim = np.loadtxt('Data/news_sim.mat')
        self.sorted_sim = np.argsort(-self.item_sim, axis=1)
        print("Sort Complish")



    def predict(self, user, item):
        # 找出K近邻项目集
        item_k = self.sorted_sim[item]
        item_topK = []
        k = 0
        for i in item_k:
            if k >= self.knn:
                break
            if self.ui_mat[user, i] > 0:
                item_topK.append(i)
                k += 1

        if len(item_topK) == 0 or self.item_sim[item_topK[0], item] < 1e-10:
            return 0

        prediction = 0
        sim_sum = 0
        for ik in item_topK:
            sim_sum += self.item_sim[item, ik]
            # uki = self.ui_mat[user, ik]
            # prediction += self.item_sim[item, ik] * uki

        return sim_sum
        if sim_sum < 1e-10:
            prediction = 0
        else:
            prediction = prediction / sim_sum
        return prediction


# def cal_sim():
#     tfidf_mat = np.loadtxt('Data/tfidf.mat')
#     print(np.shape(tfidf_mat))
#     item_sim = cosine_similarity(tfidf_mat)
#     print("Sim Complish")
#     np.savetxt('Data/item_sim.mat', item_sim)
