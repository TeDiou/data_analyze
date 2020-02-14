import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing
import numpy as np
import PIL.Image as image
from skimage import color

# 加载图像，并对数据进行规范化
def load_data(filePath):
    # 读文件
    f = open(filePath,'rb')
    data = []
    # 得到图像的像素值
    img = image.open(f)
    # 得到图像尺寸
    width, height = img.size
    for x in range(width):
        for y in range(height):
            # 得到点(x,y)的三个通道值
            c1, c2, c3 = img.getpixel((x, y))
            data.append([c1, c2, c3])
    f.close()
    # 采用Min-Max规范化
    mm = preprocessing.MinMaxScaler()
    data = mm.fit_transform(data)
    return data, width, height


# 加载图像，得到规范化的结果img，以及图像尺寸
img, width, height = load_data(r'D:\data4analyze\baby.jpg')

# 用K-Means对图像进行16聚类
kmeans =KMeans(n_clusters=16)
kmeans.fit(img)
label = kmeans.predict(img)
# 将图像聚类结果，转化成图像尺寸的矩阵

label = label.reshape([width, height])
# 将聚类标识矩阵转化为不同颜色的矩阵
label_color = (color.label2rgb(label)*255).astype(np.uint8)
label_color = label_color.transpose(1,0,2)
images = image.fromarray(label_color)
images.save('weixin_mark_color.jpg')
# 用K-Means对图像进行2聚类
kmeans =KMeans(n_clusters=2)
kmeans.fit(img)
label = kmeans.predict(img)
# 将图像聚类结果，转化成图像尺寸的矩阵
label = label.reshape([width, height])
# 创建个新图像pic_mark，用来保存图像聚类的结果，并设置不同的灰度值
pic_mark = image.new("L", (width, height))
for x in range(width):
    for y in range(height):
        # 根据类别设置图像灰度, 类别0 灰度值为255， 类别1 灰度值为127
        pic_mark.putpixel((x, y), int(256/(label[x][y]+1))-1)
pic_mark.save("weixin_mark.jpg", "JPEG")


# 载入足球数据
# data = pd.read_csv(r'D:\data4analyze\asia_soccerteam_rank_data.csv',
#                    encoding='GBK')
# data_X = data[['2019年国际排名', '2018世界杯', '2015亚洲杯']]
# # Minmax规范化到 [0,1] 空间
# mm = preprocessing.MinMaxScaler()
# data_X = mm.fit_transform(data_X)
# # 模型建立 3个聚类
# km_model = KMeans(n_clusters=3)
# km_model.fit(data_X)
# pred = km_model.predict(data_X)
# print(pred)
# pred = np.array([i+1 for i in pred])
# data = pd.concat((data, pd.DataFrame(pred)), axis=1)
# data.rename(columns={0: '聚类'}, inplace=True)
# print(data)
