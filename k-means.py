import matplotlib.pyplot as plt
import pandas as pd
from alrogrithm import bayes
from sklearn.cluster import KMeans

def getData():
    train_df = pd.read_csv('adult.csv')

    # 構造0，1变量
    age = pd.get_dummies(train_df['age'])
    # print('age：')
    # print(age)
    education = pd.get_dummies(train_df['education'])
    # print('education:')
    # print(education)
    marital_status = pd.get_dummies(train_df['marital.status'])
    # print('marital_tatus:')
    # print(marital_status)
    race = pd.get_dummies(train_df['race'])
    # print('race:')
    # print(race)
    sex = pd.get_dummies(train_df['sex'])
    # print('marital_tatus:')
    # print(marital_status)

    # 合并训练集
    train_set = pd.concat([age, education, marital_status, race], axis=1)
    return train_set


def kmeans():
    train_set=getData()
    mod = KMeans(n_clusters=2)
    y_pred=mod.fit_predict(train_set)


    r1 = pd.Series(mod.labels_).value_counts()
    # print(r1)
    r2=  pd.DataFrame(mod.cluster_centers_)
    # print(r2)
    r = pd.concat([r2, r1], axis = 1)
    r.columns = list(train_set.columns) + [u'类别数目']
    # print(r)

    #给每一条数据标注上被分为哪一类
    r = pd.concat([train_set,pd.Series(mod.labels_,index=train_set.index)],axis=1)
    r.columns = list(train_set.columns) + [u'聚类类别']
    print(r)

kmeans()