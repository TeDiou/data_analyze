import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score

#读取文件
data = pd.read_csv(r'D:\data4analyze\data_breastcaner.csv')

#分3组
features_mean = list(data.columns[2:12])
features_se = list(data.columns[12::22])
features_worst = list(data.columns[22:32])

#清洗
data.drop('id',1,inplace=True)
data['diagnosis'] = data['diagnosis'].map({'M': 1,'B': 0})

# 特征选择
features_remain = data.columns[1:31]
x = data[features_remain]
y = data.diagnosis
# 抽取30%的数据作为测试集，其余作为训练集
train, test = train_test_split(data, test_size=0.3)
# 抽取特征选择的数值作为训练和测试数据
train_X = train[features_remain]
train_y = train['diagnosis']
test_X = test[features_remain]
test_y = test['diagnosis']

# 采用Z-Score规范化数据，保证每个特征维度的数据均值为0，方差为1
ss = preprocessing.StandardScaler()
train_X = ss.fit_transform(train_X)
test_X = ss.transform(test_X)
x = ss.transform(x)

# 创建SVM分类器
model = svm.LinearSVC(max_iter=1000)
# 用训练集做训练
model.fit(train_X, train_y)
# 用测试集做预测
prediction = model.predict(test_X)
print('准确率: ', metrics.accuracy_score(prediction, test_y))
# 使用 K 折交叉验证 统计决策树准确率

print(u'cross_val_score 准确率为 %.4lf' %
      np.mean(cross_val_score(model, x, y, cv=10)))

print('mae: %.4lf' % metrics.mean_absolute_error(test_y, prediction))

