# CarefulRubbishMessage

## 项目简介

**CarefulRubbishMessage（小心垃圾短信）** 是一个基于自然语言处理与机器学习的中文垃圾短信识别系统。项目面向中文短信文本分类任务，能够对用户输入的短信内容进行自动判断，识别其是否属于垃圾短信。

本项目完成了从数据预处理、中文分词、特征提取、模型训练、模型评估到 Web 页面交互展示的完整流程。系统支持多种机器学习分类模型对比，包括多项式朴素贝叶斯、逻辑回归和支持向量机，并提供 Flask Web 界面，方便用户输入短信文本并查看预测结果。

## 项目功能

* 中文短信文本预处理
* URL、HTML 标签、特殊符号、数字等噪声清洗
* Jieba 中文分词
* 停用词过滤
* TF-IDF 文本特征提取
* 垃圾短信二分类预测
* 多模型训练与对比
* 网格搜索自动调参
* 模型准确率、F1 值、分类报告评估
* Flask Web 页面交互
* 支持用户输入短信并返回分类结果
* 支持查看不同模型的评估结果

## 数据集说明

项目使用课堂提供的中文短信数据集 `SMSSpam.csv`。

数据集包含正常短信和垃圾短信两类文本，标签含义如下：

```text
0：正常短信
1：垃圾短信
```

数据集规模约为 10973 条短信文本，项目中按照以下比例划分数据集：

```text
训练集：70%
测试集：30%
```

数据示例：

```text
1|张艺谋走下神坛 两妻争宠大打出手[搜狐]-中国移动冲浪助手:http://go.10086.cn/...
1|华夏学员午盘今日建仓【300223北京君正】明日盘中必将拉高5以上，可验证！
0|【华夏信用卡】您尾号5595的华夏信用卡于18:21消费人民币39.69元，欲查询可用余额，请回复YE5595。
```

## 技术栈

### 编程语言

* Python

### 数据处理

* numpy
* pandas
* re
* os
* joblib

### 中文文本处理

* jieba
* 停用词表
* 正则表达式文本清洗

### 特征工程

* TF-IDF
* TfidfVectorizer
* n-gram 特征

### 机器学习

* MultinomialNB
* LogisticRegression
* SVC
* GridSearchCV
* train_test_split
* accuracy_score
* f1_score
* classification_report

### Web 开发

* Flask
* HTML
* CSS
* JavaScript
* Tailwind CSS
* Font Awesome

## 项目流程

### 1. 数据预处理

项目首先对原始短信文本进行清洗和规范化处理，主要包括：

* 去除 URL 链接
* 去除 HTML 标签
* 去除特殊符号和非文本字符
* 去除数字
* 使用 Jieba 进行中文分词
* 去除停用词
* 过滤长度过短的词语
* 将分词结果拼接为适合向量化处理的文本格式

预处理可以减少短信中的噪声信息，提高后续模型对关键词和语义特征的学习效果。

### 2. 文本向量化

项目使用 `TfidfVectorizer` 将清洗后的文本转换为 TF-IDF 特征矩阵。

TF-IDF 可以衡量词语在当前文本中的重要程度，同时降低高频但区分度较低词语的影响。项目中通过设置 `min_df` 和 `max_df` 过滤过低频和过高频的词语，以减少噪声特征。

### 3. 模型训练

项目对比了三种常见机器学习模型：

#### MultinomialNB

多项式朴素贝叶斯模型适合文本分类任务，计算速度快，对高维稀疏文本特征表现较好。

#### LogisticRegression

逻辑回归是一种线性分类模型，适合二分类任务，可以输出分类概率，便于计算预测置信度。

#### SVM

支持向量机适合处理高维特征空间中的分类问题。项目中通过设置 `probability=True` 启用概率输出，使其能够返回预测置信度。

### 4. 网格搜索调参

项目使用 `GridSearchCV` 对不同模型进行参数搜索，通过交叉验证寻找较优参数组合。

调参内容包括：

* 朴素贝叶斯的 `alpha`
* 逻辑回归的 `C` 和 `penalty`
* SVM 的 `kernel` 等参数

通过网格搜索可以减少人工调参的不确定性，提高模型的泛化能力。

### 5. 模型评估

项目使用以下指标评估模型表现：

* Accuracy：准确率
* F1 Macro：宏平均 F1 值
* Precision：精确率
* Recall：召回率
* Classification Report：分类报告

最终系统会比较不同模型的 F1 值，并保存综合表现较好的模型作为最佳模型。

### 6. Web 页面交互

项目使用 Flask 搭建 Web 服务，前端页面支持用户输入短信内容，并通过接口向后端发送请求。

主要接口包括：

```text
/predict
/evaluation
```

其中 `/predict` 接口用于接收用户输入的短信文本和选择的模型，后端完成文本预处理、特征转换和模型预测后，将分类结果和置信度以 JSON 格式返回前端。

## 页面模块

Web 页面主要包含以下模块：

### 导航栏

用于切换系统中的主要功能区域，包括短信分类、模型评估和关于项目等内容。

### 短信分类模块

用户可以输入一段短信文本，选择分类模型，点击分析后获得分类结果。

系统会返回：

* 是否为垃圾短信
* 使用的模型
* 预测置信度
* 结果提示信息

### 模型评估模块

用于展示不同模型在测试集上的评估结果，包括准确率、F1 值、最佳参数和模型对比结果。

### 关于模块

用于介绍系统功能、技术特点和支持的模型。

## 项目结构

项目结构可参考如下：

```text
CarefulRubbishMessage
├── app.py
├── train.py
├── predict.py
├── requirements.txt
├── README.md
├── data
│   └── SMSSpam.csv
├── model
│   ├── best_model.pkl
│   ├── vectorizer.pkl
│   └── model_results.pkl
├── templates
│   └── index.html
├── static
│   ├── css
│   ├── js
│   └── images
└── utils
    ├── preprocess.py
    └── model_utils.py
```

如果实际项目目录与上述结构不完全一致，请以实际代码文件为准。

## 安装与运行

### 1. 克隆项目

```bash
git clone git@github.com:liu23377777/CarefulRubbishMessage.git
cd CarefulRubbishMessage
```

### 2. 创建虚拟环境

Windows：

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS / Linux：

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

如果项目中还没有 `requirements.txt`，可以根据实际使用的库生成：

```bash
pip freeze > requirements.txt
```

### 4. 训练模型

```bash
python train.py
```

训练完成后，模型文件会保存到指定目录中，例如 `model/` 或 `models/`。

### 5. 启动 Web 服务

```bash
python app.py
```

启动成功后，在浏览器中访问：

```text
http://127.0.0.1:5000
```

即可进入垃圾短信识别系统页面。

## 使用示例

输入短信内容：

```text
恭喜您获得免费提现额度，点击链接立即领取高额奖励！
```

系统可能返回：

```text
分类结果：垃圾短信
预测模型：LogisticRegression
置信度：较高
```

输入正常短信内容：

```text
【银行通知】您尾号5595的信用卡于18:21消费人民币39.69元。
```

系统可能返回：

```text
分类结果：正常短信
预测模型：MultinomialNB
置信度：较高
```

## 模型说明

| 模型                 | 特点              | 适用场景        |
| ------------------ | --------------- | ----------- |
| MultinomialNB      | 训练速度快，对文本稀疏特征友好 | 基础文本分类      |
| LogisticRegression | 线性模型，可输出概率      | 二分类任务       |
| SVM                | 适合高维特征，分类能力较强   | 文本分类与复杂边界分类 |

## 项目亮点

* 实现了完整的 NLP 文本分类流程
* 支持中文短信清洗、分词和停用词处理
* 使用 TF-IDF 提取文本特征
* 对比了多种机器学习分类模型
* 使用 GridSearchCV 自动搜索较优参数
* 提供模型评估结果展示
* 使用 Flask 实现 Web 交互界面
* 前端页面支持异步请求和动态展示预测结果

## 遇到的问题与解决方法

### 1. 文本中存在特殊符号和乱码

短信文本中可能包含 URL、HTML 标签、表情符号、特殊字符和乱码。项目通过正则表达式对文本进行清洗，提高文本质量。

### 2. SVM 默认无法输出概率

SVC 默认 `probability=False`，直接调用 `predict_proba()` 会报错。项目中将其设置为：

```python
SVC(probability=True)
```

从而支持输出预测概率和置信度。

### 3. 模型可能出现过拟合或欠拟合

项目使用 `GridSearchCV` 对模型参数进行网格搜索，并结合交叉验证选择较优参数，提升模型泛化能力。

## 后续优化方向

* 引入 BERT、RoBERTa 等预训练语言模型
* 使用 Word2Vec、FastText 等词向量方法增强文本表示
* 增加更丰富的垃圾短信数据集
* 优化前端交互体验
* 增加用户反馈和主动学习机制
* 支持模型在线更新
* 将模型部署到移动端或接口服务中
* 增加垃圾短信自动拦截功能

## 参考资料

* 自然语言处理课程资料
* 机器学习与数据挖掘相关教材
* Scikit-learn 官方文档
* Flask 框架相关教程
* Jieba 中文分词工具文档

## 项目说明

本项目为自然语言处理课程期末大作业，主要用于学习和实践中文文本分类、机器学习模型训练、模型评估以及 Web 系统集成。
