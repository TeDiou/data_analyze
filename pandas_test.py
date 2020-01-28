import pandas as pd

#data is lowa
#读入数据集
file_path = 'D:/data4analyze/train.csv'
df = pd.read_csv(file_path)
#求出需要的值和格式
avgpriceby_yearmonth = pd.DataFrame(df.groupby(['YrSold', 'MoSold']).
                                    SalePrice.mean().round())
countpriceby_yearmonth = pd.DataFrame(df.groupby(['YrSold', 'MoSold']).
                                      SalePrice.count().round())
#生成统计表
pricetable = avgpriceby_yearmonth.merge(countpriceby_yearmonth,
                                        on=["YrSold", "MoSold"])
pricetable = pricetable.rename(index=str, columns={'SalePrice_x': 'AvgSalePrice',
                                        'SalePrice_y' : 'SaleCount'})
pricetable.reset_index(inplace=True)
pricetable['Period'] = pd.to_datetime(pricetable.YrSold.astype(str) +
                                      pricetable.MoSold.astype(str),errors='ignore',format='%Y%m')
pricetable['Period'] = pricetable['Period'].dt.strftime("%Y-%m")
import matplotlib.pyplot as plt

pricetable.plot.bar(x="Period",y="SaleCount", figsize=(20, 10))
plt.xticks(rotation=45)
plt.title("SaleCount by Year and Month", fontsize = 20)
plt.show()
print(pricetable.shape)
