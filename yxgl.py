import numpy as np
import jieba
import pandas as pd
import wordcloud
from scipy.misc import imread
import matplotlib.pyplot as plt
from pylab import mpl
import seaborn as sns
import codecs
#指定默认字体
mpl.rcParams["font.sans-serif"]=["SimHei"]
mpl.rcParams["axes.unicode_minus"]
#导入停用词
stop_list = pd.read_csv("./stop_word.txt",encoding="utf-8")
stop_list=stop_list.columns[1].split()
#导入自定义词库，用load_userdict()方法批量导入
jieba.load_userdict("./yxgl_word.txt")
#导入延禧攻略文档，进行jieba分词，并进行过滤
yxgl = pd.read_csv("./yxgl.txt",encoding="utf-8",names=["content"])
yxgl_cut= []
for t in yxgl.index:
    yxgl_cut.extend(list(jieba.cut(yxgl["content"][t])))
yxgl_cut = [t for t in yxgl_cut if t not in stop_list and len(t)>1 ]

#选取词频前20
yxgl_count = pd.Series(yxgl_cut).value_counts()[0:20]

fig = plt.figure(figsize=(15,8))
x = yxgl_count.index.tolist()
y = yxgl_count.values.tolist()
sns.barplot(x,y)
plt.title("延禧攻略词频TOP20")
plt.ylabel("词频")
sns.despine(bottom=True)#去除边框
plt.savefig("./Top20.jpg",dpi=400)
plt.show()

fig =plt.figure(figsize=(15,5))
#将词频映射到词云上，设定背景轮廓
yxgl_background=imread('./background.png')#背景图片
yxgl_backcolor = imread("./backcolor.jpg")#对词语的颜色做些美化
yxgl_cloud = wordcloud.WordCloud(font_path='./simhei.ttf',mask = yxgl_background, background_color="white",
                               color_func=wordcloud.ImageColorGenerator(yxgl_backcolor) ).generate(' '.join(yxgl_cut))
#用matplotlib，进行图像展示
plt.imshow(yxgl_cloud)
plt.axis("off")#不显示坐标轴
plt.show()
plt.savefig("./yxgl_cloud.png")

#显示人物关系图
relation ={};lineNames= []

#导入任务名字文件
people = pd.read_csv("./yxgl_word.txt",encoding="utf-8",names=["name"])['name'].tolist()
people=[n.split(" ")[0] for n in people]

#记录小说每行出现的人物名字
with codecs.open("./yxgl.txt","r","utf8") as f:
    for line in f.readlines():
        poss = jieba.cut(line)
        lineNames.append([])
        for ele in poss:
            if ele not in people:
                continue
            lineNames[-1].append(ele)
            if relation.get(ele) is None:
                relation[ele] ={}
#统计人物的关系，同一行出现的关系加1
for line in lineNames:
    for name1 in line:
        for name2 in line:
            if name1==name2:
                continue
            if relation[name1].get(name2) is None:
                relation[name1][name2] =1
            else:
                relation[name1][name2]+=1

gephi_edge = pd.DataFrame(columns=['Source','Target','Weight'])
#保存关系大于10次的数据
for name3,edges in relation.items():
    for name4, relation_time in edges.items():
        if relation_time > 10:
            gephi_edge.loc[len(gephi_edge)] = [name3,name4,relation_time]
            
#保存人物联系关系
gephi_edge.to_csv('./gephi_edge1.csv',index=0)