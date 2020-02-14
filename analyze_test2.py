from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
import itertools

def plot_confusion_matrix(cm, classes, normalize=False,
                          title='Confusion matrix"', cmap=plt.cm.Blues):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def show_metrics():
    tp = cm[1, 1]
    fn = cm[1, 0]
    fp = cm[0, 1]
    tn = cm[0, 0]
    print('精确率: {:.3f}'.format(tp/(tp+fp)))
    print('召回率: {:.3f}'.format(tp/(tp+fn)))
    print('F1 值: {:.3f}'.format(2*(((tp/(tp+fp))*(tp/(tp+fn)))/((tp/(tp+fp))+(tp/(tp+fn))))))


# 绘制精确率 - 召回率曲线
def plot_precision_recall():
    plt.step(recall, precision, color = 'b', alpha = 0.2, where = 'post')
    plt.fill_between(recall, precision, step ='post', alpha = 0.2, color = 'b')
    plt.plot(recall, precision, linewidth=2)
    plt.xlim([0.0,1])
    plt.ylim([0.0,1.05])
    plt.xlabel('召回率')
    plt.ylabel('精确率')
    plt.title('精确率 - 召回率 曲线')
    plt.show();
# 加载数据
pd.set_option('display.max_column', None)
data = pd.read_csv(r'D:\data4analyze\credit_fraud\creditcard.csv')
# 数据探索
print(data.head())
# print(data.Class.value_counts())
plt.figure(figsize=(14, 14))
plt.rcParams['font.sans-serif'] = ['SimHei']
# features = ['Time', 'Amount', 'Class']
# data_corr = data[features]
# corr = data_corr.corr()
# sns.heatmap(corr, annot=True)
# plt.show()
# f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 8),)
# bins = 50
# ax1.hist(data.Time[data.Class == 1], bins=bins, color='deeppink')
# ax1.set_title('诈骗交易')
# ax2.hist(data.Time[data.Class == 0], bins=bins, color='deepskyblue')
# ax2.set_title('正常交易')
# plt.xlabel('时间')
# plt.ylabel('交易次数')
# plt.show()
# 数据规范化
data['Amount'] = preprocessing.StandardScaler().\
    fit_transform(data.Amount.values.reshape(-1, 1))
X = data[data.columns[1:-1]]
y = data[data.columns[-1]]
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=1)
svc_m = SVC(random_state=1)
svc_m.fit(train_X, train_y)
predict_y = svc_m.predict(test_X)
score_y = svc_m.decision_function(test_X)
print(score_y, predict_y)
cm = confusion_matrix(test_y, predict_y)
# 显示混淆矩阵
plot_confusion_matrix(cm, classes=[0, 1], title='逻辑回归 混淆矩阵')
# 显示模型评估分数
show_metrics()
precision, recall, thresholds = precision_recall_curve(test_y, score_y)
plot_precision_recall()
