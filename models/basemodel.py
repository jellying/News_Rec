"""Basemodel 由所有模型继承重写关键方法"""
import numpy as np
import math
from scipy.sparse import csr_matrix
import random
class basemodel():
    def __init__(self, user_news_df):
        self.ui_df = user_news_df
        self.USER_NUM = 10000 #10000  ml : 943
        self.ITEM_NUM = 6183 #6183  ml : 1682
        self.user_flag = [False for _ in range(self.USER_NUM)]
        self.ui_mat = self.get_mat(user_news_df)
        self.item_clicksum = np.sum(self.ui_mat, axis=0)
        self.item_clicksum = np.argsort(-self.item_clicksum)

    def get_mat(self, ui_df):
        read_sum = ui_df.shape[0]
        user_row = np.array([self.ui_df.iloc[i, 0] for i in range(read_sum)])
        item_col = np.array([self.ui_df.iloc[i, 1] for i in range(read_sum)])
        mat = np.zeros((self.USER_NUM, self.ITEM_NUM))
        for i in range(read_sum):
            self.user_flag[user_row[i]] = True
            mat[user_row[i], item_col[i]] = 1
        return mat

    def train(self):
        pass

    def predict(self, user, item):
        """预测user对item的评分"""
        prediction = self.item_clicksum[item]
        return prediction

    def predict_topK(self, user, K):
        """生成给用户user推荐的K个项目列表"""
        if self.user_flag[user] == False:
            return self.item_clicksum[0:K]
            # return random.sample(list(self.item_clicksum[0: 100]), K)
        user_rating = self.ui_mat[user, :]
        rec_list = dict()
        # k = 0
        for item in range(self.ITEM_NUM):
            # print(k)
            # k += 1
            if user_rating[item] == 0:
                rec_list[item] = self.predict(user, item)
        rec_topK = sorted(rec_list.items(), key=lambda e: e[1], reverse=True)
        return [rec_topK[i][0] for i in range(K)]

    def evaluation(self, test_df, topn = 10):
        read_sum = test_df.shape[0]
        user_row = np.array([test_df.iloc[i, 0] for i in range(read_sum)])
        item_col = np.array([test_df.iloc[i, 1] for i in range(read_sum)])
        read_score = np.array([1 for i in range(read_sum)])
        self.test_mat = csr_matrix((read_score, (user_row, item_col)), shape=(self.USER_NUM, self.ITEM_NUM))
        ui_dict = dict()
        for i in range(test_df.shape[0]):
            if test_df.iloc[i, 0] not in ui_dict.keys():
                ui_dict[test_df.iloc[i, 0]] = [test_df.iloc[i, 1]]
            else:
                ui_dict[test_df.iloc[i, 0]].append(test_df.iloc[i, 1])
        # 计算MAP和NDCG
        mAP = 0
        mPrecision = 0
        eval_user = 0
        user_sum = len(ui_dict)
        each = user_sum / 4
        start = each * 0
        end = each * 1
        for user, itemlist in ui_dict.items():
            # if self.user_flag[user] == False:
            #     user_sum -= 1
            #     continue
            eval_user += 1
            if eval_user % 1 == 0:
                print("Eval process: %d / %d" % (eval_user, user_sum))
            if eval_user > user_sum:
                break
            # if eval_user > end or eval_user <= start:
            #     continue
            predlist = self.predict_topK(user, topn)
            reclist = list(set(itemlist))
            mPrecision += self.cal_PN(predlist, reclist)
            mAP += self.cal_AP(predlist, reclist)
            # nDCG += self.cal_DCG(user, predlist, reclist)
        mPrecision /= user_sum
        mAP /= user_sum
        # nDCG /= user_sum
        print("Top%d Rec Result:" % topn)
        print("mPrecision: %g  mAP: %g" % (mPrecision, mAP))


    def cal_PN(self, predlist, reclist, n=10):
        p = 0
        for pred in predlist:
            if pred in reclist:
                p += 1
        p /= n
        return p

    def cal_AP(self, predlist, reclist):
        pos = 1
        rel = 1
        ap = 0
        for i in range(len(reclist)):
            if reclist[i] in predlist:
                ap += rel / pos
                rel += 1
            pos += 1
        ap /= len(reclist)
        return ap

    def cal_DCG(self, user, predlist, reclist, n=10):
        pred_rank = [self.test_mat[user, item] for item in predlist]
        rec_rank = [self.test_mat[user, item] for item in reclist]
        dcg = pred_rank[0]
        idcg = rec_rank[0]
        for i in range(1, len(pred_rank)):
            dcg += pred_rank[i] / math.log2(i + 1)
        for i in range(1, len(rec_rank)):
            idcg += rec_rank[i] / math.log2(i + 1)
        ndcg = dcg / idcg
        return ndcg


