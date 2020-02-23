import pandas as pd
from sklearn import *
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


#读取训练集和测试集
train_df=pd.read_csv('adult(1).csv')
# print(train_df)

#对Label进行编码
le=preprocessing.LabelEncoder()
income_size_encode=le.fit_transform(train_df['income'])

#構造0，1变量
age = pd.get_dummies(train_df['age'])
education=pd.get_dummies(train_df['education'])
marital_status=pd.get_dummies(train_df['marital.status'])
race=pd.get_dummies(train_df['race'])
sex=pd.get_dummies(train_df['sex'])

#合并训练集
train_set=pd.concat([age,education,marital_status,race,sex],axis=1)
train_set['income size']=income_size_encode

#从训练集
x=train_set.loc[:,train_set.columns!='income size']
y=train_set['income size']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=1)

#创建训练模型
model=BernoulliNB()
model.fit(x_train,y_train)

#预测
y_prediction=model.predict(x_test)

print('model accuracy:',round(metrics.accuracy_score(y_test,y_prediction),3))