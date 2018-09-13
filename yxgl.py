import numpy as np
import jieba
import pandas as pd
import wordcloud
from scipy.misc import imread
import matplotlib.pyplot as plt
from pylab import mpl
import seaborn as sns
import codecs
#ָ��Ĭ������
mpl.rcParams["font.sans-serif"]=["SimHei"]
mpl.rcParams["axes.unicode_minus"]
#����ͣ�ô�
stop_list = pd.read_csv("./stop_word.txt",encoding="utf-8")
stop_list=stop_list.columns[1].split()
#�����Զ���ʿ⣬��load_userdict()������������
jieba.load_userdict("./yxgl_word.txt")
#�������������ĵ�������jieba�ִʣ������й���
yxgl = pd.read_csv("./yxgl.txt",encoding="utf-8",names=["content"])
yxgl_cut= []
for t in yxgl.index:
    yxgl_cut.extend(list(jieba.cut(yxgl["content"][t])))
yxgl_cut = [t for t in yxgl_cut if t not in stop_list and len(t)>1 ]

#ѡȡ��Ƶǰ20
yxgl_count = pd.Series(yxgl_cut).value_counts()[0:20]

fig = plt.figure(figsize=(15,8))
x = yxgl_count.index.tolist()
y = yxgl_count.values.tolist()
sns.barplot(x,y)
plt.title("�������Դ�ƵTOP20")
plt.ylabel("��Ƶ")
sns.despine(bottom=True)#ȥ���߿�
plt.savefig("./Top20.jpg",dpi=400)
plt.show()

fig =plt.figure(figsize=(15,5))
#����Ƶӳ�䵽�����ϣ��趨��������
yxgl_background=imread('./background.png')#����ͼƬ
yxgl_backcolor = imread("./backcolor.jpg")#�Դ������ɫ��Щ����
yxgl_cloud = wordcloud.WordCloud(font_path='./simhei.ttf',mask = yxgl_background, background_color="white",
                               color_func=wordcloud.ImageColorGenerator(yxgl_backcolor) ).generate(' '.join(yxgl_cut))
#��matplotlib������ͼ��չʾ
plt.imshow(yxgl_cloud)
plt.axis("off")#����ʾ������
plt.show()
plt.savefig("./yxgl_cloud.png")

#��ʾ�����ϵͼ
relation ={};lineNames= []

#�������������ļ�
people = pd.read_csv("./yxgl_word.txt",encoding="utf-8",names=["name"])['name'].tolist()
people=[n.split(" ")[0] for n in people]

#��¼С˵ÿ�г��ֵ���������
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
#ͳ������Ĺ�ϵ��ͬһ�г��ֵĹ�ϵ��1
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
#�����ϵ����10�ε�����
for name3,edges in relation.items():
    for name4, relation_time in edges.items():
        if relation_time > 10:
            gephi_edge.loc[len(gephi_edge)] = [name3,name4,relation_time]
            
#����������ϵ��ϵ
gephi_edge.to_csv('./gephi_edge1.csv',index=0)