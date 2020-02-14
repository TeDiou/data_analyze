import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# 载入数据
pd.set_option('display.max_columns', None)
data = pd.read_csv(r'D:\data4analyze\heros.csv', encoding='Gbk')
# 数据探索
data['攻击范围'] = data['攻击范围'].map({'近战': 1, '远程': 0})
data['最大攻速'] = data['最大攻速'].apply(lambda x: float(x.strip('%')))
features = data.columns[1:-2]
data_f = data[features]
corr = data_f.corr()
#显示热力图
# 设置 plt 正确显示中文
# 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']
# 用来正常显示负号
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(14, 14))
sns.heatmap(corr, annot=True)
# plt.show()
# 筛选特征
# features = ['最大生命', '初始生命', '最大法力', '最高物攻',
#             '初始物攻', '最大物防', '初始物防', '初始每5秒回血',
#             '最大每5秒回蓝', '初始每5秒回蓝', '最大攻速', '攻击范围']
data_f = data[features]
# 数据标准化
ss = StandardScaler()
data_f = ss.fit_transform(data_f)
# 载入模型
gm_model = GaussianMixture(n_components=6)
pred = gm_model.fit_predict(data_f)
data = pd.concat((data, pd.Series(pred, name='聚类')), axis=1)

from sklearn.metrics import calinski_harabaz_score
print(calinski_harabaz_score(data_f, pred))
data.to_csv('test.csv', encoding='GBK')
print(data.groupby('聚类').sum())
