import pandas as pd
import networkx as nx
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

# 读入文件
email = pd.read_csv(r'D:\data4analyze\input\Emails.csv', encoding='UTF-8-SIG')
persons = pd.read_csv(r'D:\data4analyze\input\Persons.csv', encoding='UTF-8-SIG')
alias = pd.read_csv(r'D:\data4analyze\input\Aliases.csv', encoding='UTF-8-SIG')

# 去除发信人和收件人为空的
email = email.loc[email.MetadataTo.notnull() & email.MetadataFrom.notnull()]
print(email.shape)
# 别名字典
alias_dic = {}
for index, row in alias.iterrows():
    alias_dic[row['Alias']] = row['PersonId']
# 人名ID字典
person_dic = {}
for index, row in persons.iterrows():
    person_dic[row['Id']] = row['Name']


# 将人物别名/实名等统一化
def unify_name(name: str):
    name = str(name).lower()
    # 去掉, 和 @后面的内容
    name = name.replace(';', '')
    name = name.replace(",", "").split("@")[0]
    if name in alias_dic.keys():
        return person_dic[alias_dic[name]]
    return name


# 画网络图
def show_graph(graph, layout='spring_layout'):
    plt.figure(figsize=(14, 14))
    # 使用 Spring Layout 布局，类似中心放射状
    if layout == 'circular_layout':
        positions=nx.circular_layout(graph)
    else:
        positions=nx.spring_layout(graph)
    # 设置网络图中的节点大小，大小与 pagerank 值相关，因为 pagerank 值很小所以需要 *20000
    nodesize = [x['pagerank']*20000 for v,x in graph.nodes(data=True)]
    # 设置网络图中的边长度
    edgesize = [np.sqrt(e[2]['weight']) for e in graph.edges(data=True)]
    # 绘制节点
    nx.draw_networkx_nodes(graph, positions, node_size=nodesize, alpha=0.4)
    # 绘制边
    nx.draw_networkx_edges(graph, positions, edge_size=edgesize, alpha=0.2)
    # 绘制节点的 label
    nx.draw_networkx_labels(graph, positions, font_size=10)
    # 输出希拉里邮件中的所有人物关系图

    plt.show()

# 将列表名字统一化
to_list = email.MetadataTo.apply(unify_name)
from_list = email.MetadataFrom.apply(unify_name)
# 人物间权重
com_weight_dic = {}
for row in zip(to_list, from_list):
    if (row[0], row[1]) not in com_weight_dic:
        com_weight_dic[(row[0], row[1])] = 1
    else:
        com_weight_dic[(row[0], row[1])] += 1
# 创建有向图列表+权重graph提供参数
com_weight_list = [(key[0], key[1], com_weight_dic[key])for key in com_weight_dic.keys()]
d = nx.DiGraph()
# edges = [("A", "B"), ("A", "C"), ("A", "D"), ("B", "A"), ("B", "D"), ("C", "A"), ("D", "B"), ("D", "C")]
d.add_weighted_edges_from(com_weight_list)
pr1 = nx.pagerank(d)
nx.set_node_attributes(d, pr1, 'pagerank')
show_graph(d)

# 设置PR 阈值
pr_threshold = 0.005
small_g = d.copy()
for n, rank in d.nodes(data=True):
    if rank['pagerank'] < pr_threshold:
        small_g.remove_node(n)

show_graph(small_g)

