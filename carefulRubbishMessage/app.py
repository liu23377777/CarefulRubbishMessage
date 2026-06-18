import pandas as pd
import numpy as np
import re
import jieba
import os
import joblib
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report
import csv

app = Flask(__name__)

# 数据路径
DATA_PATH = '/Users/wukanuo/Desktop/nlp/SMSSpam.csv'
STOPWORDS_PATH = '/Users/wukanuo/Desktop/nlp/stopwords.dat'
# 模型保存路径
MODEL_DIR = 'saved_models'
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'vectorizer.pkl')
MODEL_PATHS = {
    'MultinomialNB': os.path.join(MODEL_DIR, 'multinomial_nb.pkl'),
    'LogisticRegression': os.path.join(MODEL_DIR, 'logistic_regression.pkl'),
    'SVM': os.path.join(MODEL_DIR, 'svm.pkl'),
    'BestModel': os.path.join(MODEL_DIR, 'best_model.pkl')
}
EVALUATION_PATH = os.path.join(MODEL_DIR, 'evaluation.json')

# 全局变量
vectorizer = None
models = {}
model_names = ['MultinomialNB', 'LogisticRegression', 'SVM']
best_model = None
model_evaluation = {}


def load_stopwords():
    """加载停用词表"""
    try:
        with open(STOPWORDS_PATH, 'r', encoding='utf-8') as f:
            stopwords = [line.strip() for line in f.readlines()]
        return set(stopwords)
    except Exception as e:
        print(f"加载停用词表失败: {e}")
        return set()


def preprocess_text(text, stopwords):
    #未处理前的文本：

    """文本预处理"""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'<.*?>', '', text)       #去除 HTML 标签
    text = re.sub(r'[^\w\s]', '', text)     #保留字母、数字、空格，去除特殊符号
    text = re.sub(r'\d+', '', text)         #去除数字
    words = jieba.lcut(text)
    words = [word for word in words if word not in stopwords and len(word) > 1]
    return ' '.join(words)


def save_models():
    """保存训练好的模型和评估结果"""
    # 创建保存目录
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 保存vectorizer
    joblib.dump(vectorizer, VECTORIZER_PATH)

    # 保存各个模型
    for name, model in models.items():
        joblib.dump(model, MODEL_PATHS[name])

    # 保存最佳模型
    joblib.dump(best_model, MODEL_PATHS['BestModel'])

    # 保存评估结果
    pd.DataFrame(model_evaluation).to_json(EVALUATION_PATH)

    print(f"模型和评估结果已保存到 {MODEL_DIR} 目录")


def load_models():
    """加载已保存的模型和评估结果"""
    global vectorizer, models, best_model, model_evaluation

    try:
        # 加载vectorizer
        vectorizer = joblib.load(VECTORIZER_PATH)

        # 加载各个模型
        for name in model_names:
            models[name] = joblib.load(MODEL_PATHS[name])

        # 加载最佳模型
        best_model = joblib.load(MODEL_PATHS['BestModel'])

        # 加载评估结果
        model_evaluation = pd.read_json(EVALUATION_PATH).to_dict()

        print(f"已从 {MODEL_DIR} 目录加载模型和评估结果")
        return True
    except Exception as e:
        print(f"加载模型失败: {e}")
        return False


def train_models():
    """训练所有模型"""
    global vectorizer, models, best_model, model_evaluation

    # 加载数据
    try:
        sms = pd.read_csv(DATA_PATH, encoding='utf8', header=None, sep='|', quoting=csv.QUOTE_NONE,
                          names=['label', 'message'])
    except Exception as e:
        return f"加载数据失败: {e}"

    # 加载停用词
    stopwords = load_stopwords()

    # 预处理文本
    sms['message'] = sms['message'].astype(str)
    sms['processed_message'] = sms['message'].apply(lambda x: preprocess_text(x, stopwords))

    # 划分数据集
    X_train, X_test, Y_train, Y_test = train_test_split(
        sms['processed_message'], sms['label'], test_size=0.3, random_state=1)

    # 特征提取
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.9
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # 定义模型及其参数空间
    model_configs = {
        'MultinomialNB': {
            'model': MultinomialNB(),
            'params': {'alpha': [0.1, 0.5, 1.0]}
        },
        'LogisticRegression': {
            'model': LogisticRegression(max_iter=1000),
            'params': {'C': [0.1, 1, 10], 'penalty': ['l2']}
        },
        'SVM': {
            'model': SVC(probability=True),
            'params': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
        }
    }

    # 训练并评估模型
    best_f1 = 0

    for name, config in model_configs.items():
        print(f"训练模型: {name}")
        grid_search = GridSearchCV(
            config['model'],
            config['params'],
            cv=3,
            n_jobs=-1,
            scoring='f1_macro'
        )
        grid_search.fit(X_train_tfidf, Y_train)
        best_estimator = grid_search.best_estimator_
        y_pred = best_estimator.predict(X_test_tfidf)

        accuracy = accuracy_score(Y_test, y_pred)
        f1 = f1_score(Y_test, y_pred, average='macro')
        report = classification_report(Y_test, y_pred)

        models[name] = best_estimator
        model_evaluation[name] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'best_params': grid_search.best_params_,
            'report': report
        }
        if f1 > best_f1:
            best_f1 = f1
            best_model = best_estimator

    # 保存模型和评估结果
    save_models()

    print("模型训练完成")
    return "模型训练成功"


@app.route('/')
def index():
    """主页面"""
    return render_template('index.html', model_names=model_names)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        text = request.form['text']
        model_name = request.form['model']

        if not text or model_name not in models:
            return jsonify({'error': '参数错误'}), 400

        # 预处理文本
        stopwords = load_stopwords()
        processed_text = preprocess_text(text, stopwords)

        # 特征提取
        text_tfidf = vectorizer.transform([processed_text])

        # 预测
        model = models[model_name]
        prediction = model.predict(text_tfidf)[0]

        # 获取概率（如果模型支持）
        confidence = 100
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(text_tfidf)[0]
            confidence = max(probabilities) * 100

        result = "垃圾短信" if prediction == 1 else "正常短信"

        return jsonify({
            'result': result,
            'confidence': confidence,
            'model': model_name
        })

    except Exception as e:
        # 打印详细的错误信息，帮助调试
        print(f"预测过程中出现错误: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/evaluation', methods=['GET'])
def evaluation():
    """获取模型评估结果"""
    return jsonify(model_evaluation)


if __name__ == '__main__':
    # 尝试加载已保存的模型
    if not load_models():
        # 如果加载失败，则训练新模型
        status = train_models()
        print(status)

    # 启动Web服务器
    app.run(debug=False)


# def demo_preprocessing():
#     """演示数据预处理和TF-IDF向量化效果"""
#     # 加载数据（与训练时一致）
#     try:
#         sms = pd.read_csv(DATA_PATH, encoding='utf8', header=None, sep='|', quoting=csv.QUOTE_NONE,
#                           names=['label', 'message'])
#     except Exception as e:
#         print(f"加载数据失败: {e}")
#         return
#
#     # 加载停用词
#     stopwords = load_stopwords()
#
#     # 预处理前5条数据
#     sample_size = 5
#     samples = sms.head(sample_size)
#
#     print(f"\n===== 预处理前数据（前{sample_size}条） =====")
#     print(samples[['label', 'message']].to_string(index=False))
#
#     # 应用预处理
#     samples['processed_message'] = samples['message'].apply(lambda x: preprocess_text(x, stopwords))
#
#     print(f"\n===== 预处理后数据（前{sample_size}条） =====")
#     print(samples[['label', 'processed_message']].to_string(index=False))
#
#     # ===== 新增：加载vectorizer并展示TF-IDF矩阵 =====
#     try:
#         # 加载训练好的vectorizer
#         vectorizer = joblib.load(VECTORIZER_PATH)
#         print(f"\n===== TF-IDF向量化结果（前{sample_size}条） =====")
#
#         # 对预处理后的文本进行向量化
#         tfidf_matrix = vectorizer.transform(samples['processed_message'])
#
#         # 将稀疏矩阵转换为密集矩阵（仅用于小样本展示）
#         tfidf_dense = tfidf_matrix.toarray()
#
#         # 获取特征名称（词汇表）
#         feature_names = vectorizer.get_feature_names_out()
#
#         # 创建DataFrame展示TF-IDF值
#         tfidf_df = pd.DataFrame(tfidf_dense, columns=feature_names)
#
#         # 仅展示非零值的特征，避免矩阵过大
#         non_zero_features = {}
#         for i, row in enumerate(tfidf_dense):
#             non_zero_indices = np.nonzero(row)[0]
#             non_zero_features[f"样本{i + 1}"] = {
#                 feature_names[idx]: round(row[idx], 3)
#                 for idx in non_zero_indices
#             }
#
#         # 打印每个样本的非零TF-IDF特征
#         for sample, features in non_zero_features.items():
#             print(f"\n{sample}的非零TF-IDF特征:")
#             print(features)
#
#         # 打印词汇表大小
#         print(f"\n词汇表大小: {len(feature_names)}")
#
#     except Exception as e:
#         print(f"加载vectorizer或生成TF-IDF矩阵失败: {e}")

# 在程序入口处添加调用（仅在需要时运行）
if __name__ == '__main__':
    # 尝试加载已保存的模型
    if not load_models():
        status = train_models()
        print(status)

    # ===== 演示预处理（可注释掉，避免每次启动都运行） =====
    # demo_preprocessing()

    # 启动Web服务器
    app.run(debug=False)