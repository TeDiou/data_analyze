from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# 加载数据
digits = load_digits()
data = digits.data
# 数据探索
# print(data.shape)
# 查看第一幅图像
# print(digits.images[0])
# 第一幅图像代表的数字含义
# print(len(digits.target))
# 将第一幅图像显示出来

# 分离数据
train_x, test_x, train_y, test_y = \
    train_test_split(data, digits.target,
                     test_size=0.25, random_state=1)
# 数据规范化
ss = preprocessing.StandardScaler()
train_x = ss.fit_transform(train_x)
test_x = ss.transform(test_x)

mm = preprocessing.MinMaxScaler()
train_mm_x = mm.fit_transform(train_x)
test_mm_x = mm.transform(test_x)
# knn建立模型
knn_model = KNeighborsClassifier()
knn_model.fit(train_x, train_y)
# 预测
predict = knn_model.predict(test_x)
# svm
svm = SVC()
svm.fit(train_x, train_y)
svm_pred = svm.predict(test_x)
# bayes
nb = MultinomialNB()
nb.fit(train_mm_x, train_y)
nb_pred = nb.predict(test_mm_x)
# Decision tree
dtc = DecisionTreeClassifier()
dtc.fit(train_x, train_y)
dtc_pred = dtc.predict(test_x)
# Random Forest
rf = RandomForestClassifier()
rf.fit(train_x, train_y)
rf_pred = rf.predict(test_x)

print('KNN预测结果: {}'.format(accuracy_score(test_y, predict)))
print('SVM预测结果: {}'.format(accuracy_score(test_y, svm_pred)))
print('Naive bayes预测结果: {}'
      .format(accuracy_score(test_y, nb_pred)))
print('Decision tree预测结果: {}'
      .format(accuracy_score(test_y, dtc_pred)))
print('Random Forest预测结果: {}'
      .format(accuracy_score(test_y, rf_pred)))
print('Random Forest预测结果: {}'
      .format(rf.score(test_x, test_y)))
