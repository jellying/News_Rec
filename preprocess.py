'''
对数据集的预处理，包括统计平均用户点击，新闻被点击次数
抽取用户-新闻对数据，和新闻id-内容数据
根据时间戳分割训练集和测试集
'''
import pandas as pd
import time
import re
from gensim.models import ldamodel
from gensim import corpora
import jieba
from jieba import analyse
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jieba.posseg
import json

# 统计用户点击情况
def statistic():
    data_df = pd.read_csv("Data/user_click_data.txt", sep='\t', header=-1)
    user_dict = dict()
    news_dict = dict()
    for i in range(data_df.shape[0]):
        user_id = data_df.iloc[i, 0]
        news_id = data_df.iloc[i, 1]
        if user_id not in user_dict:
            user_dict[user_id] = 1
        else:
            user_dict[user_id] += 1
        if news_id not in news_dict:
            news_dict[news_id] = 1
        else:
            news_dict[news_id] += 1
    user_read_mean = sum(user_dict.values()) / len(user_dict)
    news_read_mean = sum(news_dict.values()) / len(news_dict)
    min_userread = min(user_dict.values())
    min_newsread = min(news_dict.values())
    print("Mean user read times: %g" % user_read_mean)
    print("Mean news read times: %g" % news_read_mean)
    print("Min user read times: %d" % min_userread)
    print("Min news read times: %d" % min_newsread)

# 抽取新闻内容
def extract():
    data_df = pd.read_csv("Data/user_click_data.txt", sep='\t', header=-1)
    user_dict = dict()
    news_dict = dict()
    uid = 0
    nid = 0
    # 提取用户和新闻id,存储新闻内容
    news_info_file = open("Data/news_context.txt", 'w', encoding='utf-8')
    for i in range(data_df.shape[0]):
        user_id = data_df.iloc[i, 0]
        news_id = data_df.iloc[i, 1]
        if user_id not in user_dict:
            user_dict[user_id] = uid
            uid += 1
        if news_id not in news_dict:
            news_dict[news_id] = nid
            context = [str(nid), str(data_df.iloc[i, 3]), str(data_df.iloc[i, 4]), str(data_df.iloc[i, 5])]
            news_info_file.write('\t'.join(context) + '\n')
            nid += 1
    news_info_file.close()

    # 存储用户id字典和新闻id字典
    user_list = sorted(user_dict.items(), key=lambda e: e[1], reverse=False)
    news_list = sorted(news_dict.items(), key=lambda e: e[1], reverse=False)
    user_file = open("Data/user_id.txt", 'w', encoding='utf-8')
    for user in user_list:
        user_file.write(str(user[1]) + '\t' + str(user[0]) + '\n')
    user_file.close()
    news_file = open("Data/news_id.txt", 'w', encoding='utf-8')
    for news in news_list:
        news_file.write(str(news[1]) + '\t' + str(news[0]) + '\n')
    news_file.close()

    # 存储用户id-新闻id-时间数据
    user_news_df = data_df.loc[:, 0:2]
    for i in range(data_df.shape[0]):
        user_news_df.iloc[i, 0] = user_dict[user_news_df.iloc[i, 0]]
        user_news_df.iloc[i, 1] = news_dict[user_news_df.iloc[i, 1]]
    user_news_df.to_csv("Data/user_news_id.csv", sep='\t', header=False, index=False)

# 时间戳转日期
def timestamp_datatime(value):
    format = '%Y-%m-%d %H:%M'
    #format = '%Y-%m-%d %H:%M:%S'
    #value 为时间戳值,如:1460073600.0
    value = time.localtime(value)
    dt = time.strftime(format, value)
    return dt

# 按时间分割测试集和训练集
def split_data():
    data_df = pd.read_csv("Data/user_news_id.csv", header=-1, sep='\t')
    train_file = open("Data/train_data.txt", 'w', encoding='utf-8')
    test_file = open("Data/test_data.txt", 'w', encoding='utf-8')
    for i in range(data_df.shape[0]):
        read_time = data_df.iloc[i, 2]
        date = timestamp_datatime(read_time)
        day = date.split(' ')[0].split('-')[2]
        uid = str(data_df.iloc[i, 0])
        nid = str(data_df.iloc[i, 1])
        if int(day) < 20:
            train_file.write(uid + '\t' + nid + '\t' + date + '\n')
        else:
            test_file.write(uid + '\t' + nid + '\t' + date + '\n')
    train_file.close()
    test_file.close()


def get_delwords(path):
    '''获取需要剔除的词表'''
    with open(path, 'r', encoding='utf-8') as file:
        return set([line.strip('\n') for line in file])


# 把新闻内容转化为主题词
def extract_topic():
    ctx_data = pd.read_csv("Data/news_context.txt", sep='\t', header=-1)
    stop_set = get_delwords("Data/stopwords.dat")
    symbol_set = get_delwords("Data/symbol_list.txt")
    words_list = []
    for i in range(len(ctx_data)):
        news_ctx = str(ctx_data.iloc[i, 1]) + ' ' + str(ctx_data.iloc[i, 2])
        news_words = list(jieba.posseg.cut(news_ctx))
        words = list()
        for wrd in news_words:
            term = wrd.word
            flag = wrd.flag
            if term not in stop_set and term not in symbol_set:
                isnum = re.match(r'([a-z0-9A-Z%\._]+)', term)
                if isnum is None:
                    if 'n' in flag or 'v' in flag:
                        words.append(term)
        words_list.append(words)

    # 制作词典
    word_dict = corpora.Dictionary(words_list)
    word_dict.save_as_text("Data/vocab.txt")

    def get_dotcontent(content):
        str_list = []
        dotindex = -1
        while dotindex + 1 < len(content):
            dotindex = dotindex + content[dotindex + 1:].index('"') + 1
            s = dotindex
            dotindex = content[dotindex + 1:].index('"') + dotindex + 1
            e = dotindex
            str_list.append(content[s + 1:e])

        return str_list

    # 语料列表
    corpus_list = [word_dict.doc2bow(text) for text in words_list]
    # 构建lda模型
    lda = ldamodel.LdaModel(corpus=corpus_list, id2word=word_dict, num_topics=50, alpha='auto', iterations=1000)
    with open("Data/topic.txt", 'w', encoding='utf-8') as file:
        for topic in lda.show_topics(num_topics=50):
            file.write(str(topic[0]) + '\t' + ' '.join(get_dotcontent(topic[1])) + '\n')
    with open("Data/news_topic.txt", 'w', encoding='utf-8') as file:
        topic_dict = dict()
        for n in range(len(corpus_list)):
            topic = lda.get_document_topics(corpus_list[n])
            topic_dict[n] = topic
        file.write(json.dumps(topic_dict))
    lda.save("trained_model/lda_model")

# 将新闻内容生成tfidf矩阵, 直接计算并保存相似度矩阵
def gen_tfidf():
    ctx_data = pd.read_csv("Data/news_context.txt", sep='\t', header=-1)
    stop_set = get_delwords("Data/stopwords.dat")
    symbol_set = get_delwords("Data/symbol_list.txt")
    corpus = []
    for i in range(len(ctx_data)):
        print(i)
        news_ctx = str(ctx_data.iloc[i, 1]) + ' ' + str(ctx_data.iloc[i, 2])
        news_words = list(jieba.cut(news_ctx))
        words = list()
        for term in news_words:
            if term not in stop_set and term not in symbol_set:
                isnum = re.match(r'([a-z0-9A-Z%\._]+)', term)
                if isnum is None:
                    words.append(term)
        corpus.append(' '.join(words))
    CV = CountVectorizer()
    TFIDF = TfidfTransformer()
    tf_mat = CV.fit_transform(corpus)
    tfidf_mat = TFIDF.fit_transform(tf_mat)
    print(np.shape(tfidf_mat))
    # np.savetxt("Data/tfidf.mat", tfidf_mat)
    sim_mat = cosine_similarity(tfidf_mat)
    print(np.shape(sim_mat))
    np.savetxt("Data/news_sim.mat", sim_mat)
    print("Sim mat saved")

def cold_user():
    data_df = pd.read_csv("Data/train_data.txt", sep='\t', header=-1)
    test_df = pd.read_csv("Data/test_data.txt", sep='\t', header=-1)
    user_set = set()
    test_set = set()
    for i in range(len(data_df)):
        user_set.add(data_df.iloc[i, 0])
    for i in range(len(test_df)):
        test_set.add(test_df.iloc[i, 0])
    cold_sum = 0
    for user in test_set:
        if user not in user_set:
            cold_sum += 1
    print("User Cold Sum: %d" % cold_sum)

def gen_voc_content():
    vocab_file = open("Data/vocab.txt", 'r', encoding='utf-8')
    vocab = dict()
    k = 1
    for line in vocab_file:
        seq = line.split('\t')
        if len(seq) > 1:
            word = seq[1]
            if word not in vocab.keys():
                vocab[word] = k
                k += 1
    vocab_file.close()
    content_file = open("Data/news_context.txt", 'r', encoding='utf-8')
    id_content_file = open("Data/news_context_id.txt", 'w', encoding='utf-8')
    maxlen = 0
    for line in content_file:
        ctx_words = []
        seq = line.split('\t')
        news_id = seq[0]
        news_ctx = seq[1] + ' ' + seq[2]
        ctx_cut = jieba.cut(news_ctx)
        for word in ' '.join(ctx_cut).split(' '):
            if word in vocab.keys():
                ctx_words.append(str(vocab[word]))
        maxlen = max(len(ctx_words), maxlen)
        if len(ctx_words) == 0:
            ctx_words.append('0')
        id_content_file.write(str(news_id) + '\t' + ' '.join(ctx_words) + '\n')
    id_content_file.close()
    content_file.close()
    print("Maxlen: " + str(maxlen))

gen_tfidf()