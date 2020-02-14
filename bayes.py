import os
import pandas as pd
import jieba
import warnings
from sklearn import metrics

warnings.filterwarnings('ignore')
#初始化模型
from sklearn.feature_extraction.text import TfidfVectorizer
url_train = r'D:\data4analyze\text_classification-master\text classification\train'
url_test = r'D:\data4analyze\text_classification-master\text classification\test'

#中文分词
def cut_word(filepath):
    #加载文件列表
    str1 = ''
    text = open(filepath, 'r', encoding='gb18030').read()
    for word in jieba.cut(text):
        str1 += word + ' '
    return str1


#加载文件
def load_files(fileDir, label):
    os.chdir(fileDir)
    files = os.listdir()
    file_text_lst = []
    labels = []
    for file in files:
        file_text_lst.append(cut_word(fileDir+r'\\'+file))
        labels.append(label)
    return file_text_lst, labels


# 训练数据
train_words_list1, train_labels1 = load_files(url_train+r'\女性', '女性')
train_words_list2, train_labels2 = load_files(url_train+r'\体育', '体育')
train_words_list3, train_labels3 = load_files(url_train+r'\文学', '文学')
train_words_list4, train_labels4 = load_files(url_train+r'\校园', '校园')

train_words_list = train_words_list1 + train_words_list2 + train_words_list3 + train_words_list4
train_labels = train_labels1 + train_labels2 + train_labels3 + train_labels4

# 测试数据
test_words_list1, test_labels1 = load_files(url_test+r'\女性', '女性')
test_words_list2, test_labels2 = load_files(url_test+r'\体育', '体育')
test_words_list3, test_labels3 = load_files(url_test+r'\文学', '文学')
test_words_list4, test_labels4 = load_files(url_test+r'\校园', '校园')

test_words_list = test_words_list1 + test_words_list2 + test_words_list3 + test_words_list4
test_labels = test_labels1 + test_labels2 + test_labels3 + test_labels4

stop_words = open(r'D:\data4analyze\text_classification-master\text classification\stop\stopword.txt',
                  'r', encoding='utf-8').read()
# 列表头部\ufeff处理
stop_words = stop_words.encode('utf-8').decode('utf-8-sig')
# 根据分隔符分隔
stop_words = stop_words.split('\n')

# 计算单词权重
tf = TfidfVectorizer(stop_words=stop_words, max_df=0.5)
train_features = tf.fit_transform(train_words_list)
test_features = tf.transform(test_words_list)


# 多项式贝叶斯分类器
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB(alpha=0.001).fit(train_features, train_labels)
predict = clf.predict(test_features)

print(predict[:10])
# 计算准确率
print('准确率为：', metrics.accuracy_score(test_labels, predict))
