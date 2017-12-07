# clustering
* K-means 3聚类
* GMM kmean的簇心为初始簇，EM算法进行迭代。 返回一个概率矩阵，每个簇x心对改点的概率贡献
* NMI  用聚类后的标签和原始标签进行联合熵计算，然后标准化（归一化）,计算方式参考链接http://www.cnblogs.com/ziqiao/archive/2011/12/13/2286273.html
* GMM_ label只包含GMM输出的标签，GMMALL 包括原始点和标签，km同理
* GMM_OUT为GMM聚类可视化KM同理
* GMM _pr是GMM算法输出的参数
*  result是比较的聚类结果直接的差异数
