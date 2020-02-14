from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.metrics import accuracy_score

pd.set_option('display.max_columns', None)
# load data
data = pd.read_csv(r'D:\data4analyze\UCI_Credit_Card.csv')
# 探索
# 查看数据集大小
# print(data.shape)
# 数据集概览
# print(data.describe())
next_month = data['default.payment.next.month'].value_counts()
print(next_month)
# 特征选择
X = data[data.columns[1:-1]]
y = data[data.columns[-1]]

# 30% 作为测试集，其余作为训练集
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.30, stratify=y, random_state=1)

# 构造各种分类器
classifiers = [
    SVC(random_state=1, kernel='rbf',),
    DecisionTreeClassifier(random_state=1, criterion='gini'),
    RandomForestClassifier(random_state=1, criterion='gini'),
    KNeighborsClassifier(metric='minkowski'),
    AdaBoostClassifier(random_state=1)
]

# 分类器名称
classifier_names = [
    'svc',
    'decisiontreeclassifier',
    'randomforestclassifier',
    'kneighborsclassifier',
    'adaboostclassifier'
]

# 分类器参数
classifier_param = [
    {'svc__C': [1, 5, 10], 'svc__gamma':[0.0001, 0.0005, 0.001]},
    {'decisiontreeclassifier__max_depth': range(5, 15)},
    {'randomforestclassifier__max_depth': range(5, 15)},
    {'kneighborsclassifier__n_neighbors': [4, 6, 8]},
    {'adaboostclassifier__n_estimators': [50, 100, 150]}
]


def gridSearchCV_work(pipeline, param_grid, estimator_name, train_x, train_y, test_x, test_y):
    gscv = GridSearchCV(pipeline, param_grid)
    gscv.fit(train_x, train_y)
    print('最佳得分 %.4lf' % gscv.best_score_)
    print(estimator_name + '最佳参数 ', gscv.best_params_)
    pred = gscv.predict(test_x)
    print('准确率', accuracy_score(test_y, pred))


for Classifier, Classifier_name, params in \
    zip(classifiers, classifier_names, classifier_param):
    pipeline1 = Pipeline([
        ('scaler', preprocessing.StandardScaler()),
        (Classifier_name, Classifier)
    ])
    gridSearchCV_work(pipeline1, params, Classifier_name, train_x, train_y, test_x, test_y,)