from efficient_apriori import apriori
import csv

raw_data = csv.reader(open(r'D:\data4analyze\张艺谋.csv', 'r', encoding='UTF-8'))
# 用来装演员列表 每一个为一个电影演员tuple
lst = []
for names in raw_data:
    actors = []
    for name in names:
        actors.append(name.replace(' ',''))
    lst.append(actors[1:])
print(lst)
data, rules = apriori(lst,
                      min_support=0.1, min_confidence=1)
print(data)
print(rules)
