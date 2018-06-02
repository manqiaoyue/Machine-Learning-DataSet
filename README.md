# Machine-Learning-DataSet
下面列几个公开的出租车和公共自行车的数据集出租车
北京：T-Drive trajectory data sample - Microsoft Research 
(https://www.microsoft.com/en-us/research/publication/t-drive-trajectory-data-sample/)
上海：SJTU Wireless and Sensor Network Lab  
(http://wirelesslab.sjtu.edu.cn/taxi_trace_data.html)
纽约：NYC Taxi &amp; Limousine Commission   
(http://www.nyc.gov/html/tlc/html/about/trip_record_data.shtml)
芝加哥：Chicago Taxi Data Released    
(https://digital.cityofchicago.org/index.php/chicago-taxi-data-released/)
旧金山：https://crawdad.cs.dartmouth.edu/epfl/mobility/20090224/
罗马：https://crawdad.cs.dartmouth.edu/roma/taxi/20140717/ 
公共自行车纽约：Citi Bike System Data | Citi Bike NYC   
(https://www.citibikenyc.com/system-data)      
芝加哥：Divvy System Data | Divvy Bikes     
(https://www.divvybikes.com/system-data)    
更多公共自行车数据可见：Bike Share Data Systems    
(https://github.com/BetaNYC/Bike-Share-Data-Best-Practices/wiki/Bike-Share-Data-Systems)

计算机视觉
MNIST: 最通用的健全检查。25x25 的数据集，中心化，B&W 手写数字。这是个容易的任务——
但是在 MNIST 有效，不等同于其本身是有效的。 地址：http://pjreddie.com/projects/mnist-in-csv/
CIFAR 10 & CIFAR 100: 32x32 彩色图像。虽不再常用，但还是用了一次，可以是一项有趣的健全检查。
地址：https://www.cs.toronto.edu/~kriz/cifar.html

ImageNet: 新算法实际上的图像数据集。很多图片 API 公司从其 REST 接口获取标签，这些标签被怀疑与 ImageNet 的下一级 WordNet 的 1000 个范畴很接近。 
地址：http://image-net.org/

LSUN: 场景理解具有很多辅助任务（房间布置评估、显著性预测等）和一个相关竞争。 
地址：http://lsun.cs.princeton.edu/2016/

PASCAL VOC: 通用图像分割／分类：对于构建真实世界的图像注释毫无用处，对于基线则意义重大。
地址：http://host.robots.ox.ac.uk/pascal/VOC/

SVHN: 来自谷歌街景视图（Google Street View）的门牌号数据集。把这想象成荒野之中的周期性 MNIST。 
地址：http://ufldl.stanford.edu/housenumbers/

MS COCO: 带有一个相关性竞争的通用图像理解／字幕。
地址：http://mscoco.org/

Visual Genome: 非常详细的视觉知识库，并带有 100K 图像的深字幕。 
地址：http://visualgenome.org/

Labeled Faces in the Wild:通过名称标识符，已经为被裁剪的面部区域（用 Viola-Jones）打了标签。现有人类的子集在数据集中有两个图像。
对于这里做面部匹配系统训练的人来说，这很正常。
地址：http://vis-www.cs.umass.edu/lfw/

自然语言 文本分类数据集
（2015 年来自 Zhang 等人）：一个用于文本分类的合 8 个数据集为 1 个的大型数据集。这些是用于新文本分类的最常被报道的基线。样本大小从 120K 到 3.6M, 问题从 2 级到 14 级。
数据集来自 DBPedia、Amazon、Yelp、Yahoo!、Sogou 和 AG。 
地址https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M

WikiText：来自由 Salesforce MetaMind 精心策划的维基百科文章中的大型语言建模语料库。
地址：http://metamind.io/research/the-wikitext-long-term-dependency-language-modeling-dataset/

Question Pairs：从包含重复／语义相似性标签的 Quora 释放出来的第一个数据集。 
地址：https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs

SQuAD: 斯坦福大学问答数据集（The Stanford Question Answering Dataset）——一个被广泛应用于问题回答和阅读理解的数据集，
其中每个问题的答案形式是文本的一个片段或碎片。 
地址：https://rajpurkar.github.io/SQuAD-explorer/

CMU Q/A Dataset: 手动生成的仿真陈述问题／回答与维基百科文章的难度评级相对应。 
地址：http://www.cs.cmu.edu/~ark/QA-data/

Maluuba Datasets: 用于状态性自然语言理解研究的人工生成的精密数据集。 
地址：https://datasets.maluuba.com/

Billion Words: 大型，有统一目标的语言建模数据集。常被用来训练诸如 word2vec 或 Glove 的分布式词表征。 
地址：http://www.statmt.org/lm-benchmark/

Common Crawl: PB 级规模的网络爬行——常被用来学习词嵌入。可从 Amazon S3 上免费获取。由于它是 WWW 的抓取，同样也可以作为网络数据集来使用。
地址：http://commoncrawl.org/the-data/

bAbi: 来自 FAIR（Facebook AI Research）的合成式阅读理解与问答数据集。
地址：https://research.fb.com/projects/babi/

The Children’s Book Test：从来自古登堡计划的童书中提取（问题+上下文，回答）组的基线。这对问题回答、阅读理解和仿真陈述查询有用。 
地址：https://research.fb.com/projects/babi/

Stanford Sentiment Treebank: 标准的情感数据集，在每一个句子解析树的节点上带有细腻的情感注解。
地址：http://nlp.stanford.edu/sentiment/code.html
20 Newsgroups: 文本分类经典数据集中的一个。通常可用作纯分类或任何 IR／索引算法的基准。
地址：http://qwone.com/~jason/20Newsgroups/

Reuters: 旧的，纯粹基于分类的数据集与来自新闻专线的文本。常用于教程。 
地址：https://archive.ics.uci.edu/ml/datasets/Reuters-21578+Text+Categorization+Collection

IMDB:一个用于二元情感分类的更旧更小的数据集。 地址：http://ai.stanford.edu/~amaas/data/sentiment/

UCI’s Spambase: 来自著名的 UCI 机器学习库较久的经典垃圾电子邮件数据集。由于数据集的策划细节，这可以是一个学习个性化过滤垃圾邮件的有趣基线。 
地址：https://archive.ics.uci.edu/ml/datasets/Spambase

语音 大多数语音识别数据集是有所有权的，这些数据为收集它们的公司带来了大量的价值，所以在这一领域里，许多可用的数据集都是比较旧的。

2000 HUB5 English: 仅仅只包含英语的语音数据，最近百度发表的论文《深度语音：扩展端对端语音识别（Deep Speech: Scaling up end-to-end speech recognition）》就是使用了该语音数据集。
地址：https://catalog.ldc.upenn.edu/LDC2002T43

LibriSpeech：包括文本和语音的有声读物数据集。它是近 500 小时由多人朗读清晰的各类有声读物数据集，且由包含文本和语音的书籍章节组织起结构。
地址：http://www.openslr.org/12/

VoxForge：带口音的语音清洁数据集，特别是对于如期望对不同口音或腔调的语音有鲁棒性需求的系统很有用。 
地址：http://www.voxforge.org/

TIMIT：只包含英语的语音识别数据集。 
地址：https://catalog.ldc.upenn.edu/LDC93S1

CHIME：包含噪声的语音识别数据集。该数据集包含真实、模拟和清洁的语音记录。实际上是记录四个说话者在四个噪声源的情况下近 9000 份记录，
模拟数据是在结合话语行为和清洁无噪语音记录的多环境下生成的。
地址：http://spandh.dcs.shef.ac.uk/chime_challenge/data.html

TED-LIUM：TED 演讲的语音转录数据集。1495 份 TED 演讲的语音记录，并且这些语音记录有对应的全文本。 
地址：http://www-lium.univ-lemans.fr/en/content/ted-lium-corpus 推荐和排序系统

Netflix Challenge：第一个主要 Kaggle 风格的数据库。因为存在隐私问题，只能非正式地获得授权。 
地址：http://www.netflixprize.com/

MovieLens：各种电影的评论数据库，通常用于基线协同过滤（collaborative filtering baselines）。
地址：https://grouplens.org/datasets/movielens/

Million Song Dataset：在 Kaggle 上大量、富元数据（metadata-rich）、开源的数据集，有利于人们试验混合推荐系统（hybrid recommendation systems）。
地址：https://www.kaggle.com/c/msdchallenge

Last.fm：音乐推荐数据集，该数据集能有权访问底层社交网络和其他元数据，而这样的数据集正对混合系统有巨大的作用。
地址：http://grouplens.org/datasets/hetrec-2011/ 网络和图表

Amazon Co-Purchasing 和 Amazon Reviews：从亚马逊以及相关产品评论数据网络爬取的如「用户买了这个同时也会买哪个」这样的语句。适合在互联网中进行推荐系统的测试。 
地址：http://snap.stanford.edu/data/#amazon 和 http://snap.stanford.edu/data/amazon-meta.html

Friendster Social Network Dataset：在 Friendster 的重心转入到游戏网站之前，这家网站发布了包含 103,750,348 个用户好友列表的匿名数据集。 
地址：https://archive.org/details/friendster-dataset-201107 地理测绘数据库

OpenStreetMap：免费许可的全球矢量数据集。其包含了旧版的美国人口统计局的 TIGER 数据。
地址：http://wiki.openstreetmap.org/wiki/Planet.osm

Landsat8：整个地球表面的卫星拍摄数据，每隔几周会更新一次。 
地址：https://landsat.usgs.gov/landsat-8

NEXRAD：多普雷达扫描的美国大气环境。
地址：https://www.ncdc.noaa.gov/data-access/radar-data/nexrad 人们常常认为解决一个数据集上的问题就相当于对产品进行了一次完整的审视。
因为我们可以使用这些数据集进行验证或证明一个概念，但是也不要忘了测试模型或原型是如何获取新的和更实际的数据来提高运算效果，获得优良产品的。
数据驱动的成功公司通常从他们收集新数据、私有数据的能力中获得力量，从而以一种具有竞争力的方式提高他们的表现。
参考链接：https://medium.com/startup-grind/fueling-the-ai-gold-rush-7ae438505bc2#.mdlhr7kod

关于汽车的数据集

首先上的是车辆检测或者定位有关的：

1、KITTI数据库：The KITTI Vision Benchmark SuiteThe KITTI Vision Benchmark Suite

2、一个面向于自动驾驶的，实际路况下的车辆检测数据集：

TME Motorway Dataset (Vehicle detection)

3、一系列来自于同一地方的数据库：布尔诺科技大学的Traffic Research组（没听说过？），可能名气比较小，但是做出了很多干货
（一篇非常有意思的文章 BoxCars: Improving Vehicle Fine-Grained Recognition using 3D Bounding Boxes  的出处）  
包括了交通卡口车辆的监控视频数据集（~200G）
地址 https://medusa.fit.vutbr.cz/traffic/research-topics/fine-grained-vehicle-recognition/boxcars-improving-vehicle-fine-grained-recognition-using-3d-bounding-boxes-in-traffic-surveillance/

再上一些车辆分类的（刚刚的BoxCar也算）：
1、北京理工大学的BIT-Vehicle：BIT-Vehicle Dataset 6种车辆外形（ Bus, Microbus, Minivan, Sedan, SUV, and Truck ）,含车辆的位置标注。
地址 http://iitlab.bit.edu.cn/mcislab/vehicledb/

2、Stanford Cars dataset：http://ai.stanford.edu/~jkrause/cars/car_dataset.html 196种make&model&year，含车辆位置标注。
地址 http://ai.stanford.edu/~jkrause/cars/car_dataset.html

3、CompCars：CompCars Dataset 应该来说是目前最大的车辆数据集了，美中不足的是需要申请。 
163 car makes with 1,716 car models, a total of 136,726 images.
地址 http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/
