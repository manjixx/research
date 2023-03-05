# 数据-机理双驱动群体热舒适模型预测

## 一、数据分析

### 1.1 实验数据处理

- 将2021年冬夏数据进行整合,保留二者共同特征 ✔️

> **查看数据分布**

- 2021年全年pmv投票值、温度与湿度分布图，判断是否符合正态分布以及查看相关性最高的5个参数 ✔️
- 2021年夏天pmv投票值、温度与湿度分布图，判断是否符合正态分布以及查看相关性最高的5个参数 ✔️
- 2021年冬天pmv投票值、温度与湿度分布图，判断是否符合正态分布以及查看相关性最高的5个参数 ✔️
- 生成每个成员全年、夏季、冬季的热舒适投票值 
- 生成每一天的热舒适投票值分布

### 1.2 数据生成

- 随机生成
- 等步长生成
  - ta
    - step: 0.5
    - summer: 20-32
    - winter: 18-30
  - hr
    - step: 1
    - summer: 60-31%
    - winter: 10-31%
  - season: winter、summer
  - Air vel: 0.1~0.35m/s step:0.05
  - col
    - summer: 0.5
    - winter: 0.818
  - met: 1.1
  - age: 20~28
  - gender: 0,1
  - height:
    - step: 1
    - 男:165~185
    - 女:155~175
  - BMI
    - step: 0.5
    - 17~26
  - weight:根据bmi与height计算

### 1.3 群体划分

- 分别基于夏季与秋季数据，根据BMI、性别，主观热偏好，主观热敏感度以及聚类算法对群体进行划分

[Pandas高级数据分析快速入门之五](https://blog.csdn.net/xiaoyw71/article/details/120094548)

## 二、算法实现

通过文献调研，目前用于热舒适分类的算法主要有：

- LR
- LDA
- Soft-SVM：
- KNN：
- [Decision Tree/Classification Tree](https://zhuanlan.zhihu.com/p/361464944?ivk_sa=1024320u):CART
- Kernel Naive Bayes/NB
- Classification Neural Networks
- Discriminate Analysis
- 集成学习(Ensemble):Boosted Trees、Random Forest、RUS Boosted Tree、SubSpace KNN、SubSpace-Discriminate

[分类算法对比](https://blog.csdn.net/ex_6450/article/details/126150464)

> **拟选用算法**

- Soft-SVM
- KNN
- 集成学习
  - Adaboost,修改初始数据集的权重
  - Random Forest 
  - Xgboost
- ANN



> **舒适区域确定**

通过控制输入数据中ta和rh的取值范围，计算每个温度区间中，人群中预测值在-0.5~0.5的占比。由此可以确定80%人群满意，50%人群满意，15%人群满意

## 二、工作计划

> 2022-02-14

- 搭建`pmv`模型，计算`baseline`
- 修改`svm`与`knn`算法接口，传入参数`x_train,y_train,x_test,y_test,c_weights`
- 针对`adaboost`算法，传入训练数据的同时，自己初始化`sample_weights`并传递给模型
- 针对`randomForest`算法，根据分类标准不同初始化`feature_sample`传递给模型

> 2022-02-17

- 将`knn`模型用`sklearn`实现

> 2022-02-20

knn报错`TypeError: '<' not supported between instances of 'str' and 'int'`

- 错误原因，模型参数`leafsize`要求为`int`，实际传参类型为`str`

完成所有模型初步测试
解决knn报错bug

> 2022-02-21

- 特征选择
- knn 距离函数选择
- knn: k值确定
- randomForest,x_feature


## 六、Kernel Naive Bayes/NB

[分类算法之朴素贝叶斯 (Naive Bayes)](https://blog.csdn.net/ex_6450/article/details/126142846?spm=1001.2101.3001.6661.1&utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-126142846-blog-125782329.pc_relevant_multi_platform_whitelistv3&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-126142846-blog-125782329.pc_relevant_multi_platform_whitelistv3&utm_relevant_index=1)

## 参考资料

[机器学习练习 Scikit-learn的介绍](https://github.com/fengdu78/WZU-machine-learning-course/blob/main/code/03-%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%BA%93Scikit-learn/ML-lesson3-Scikit-learn.ipynb)

[评价指标](https://blog.csdn.net/hfutdog/article/details/88085878)

![](https://img-blog.csdnimg.cn/61fc7a3082f44882ae4d8b9d29f821e2.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAfumjjuWHjOWkqeS4i34=,size_20,color_FFFFFF,t_70,g_se,x_16)
