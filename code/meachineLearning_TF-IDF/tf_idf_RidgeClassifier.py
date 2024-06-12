from sklearn.feature_extraction.text import  TfidfVectorizer,CountVectorizer
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score

# 1. 读取数据，数据概览
x_train = pd.read_csv('D:/wk/data/personal-NLP-learning/train_set.csv',sep='\t') # ,nrows=15000
x_test = pd.read_csv('D:/wk/data/personal-NLP-learning/test_a.csv', sep = '\t') # ,nrows=15000


print(x_train.head())
print('----------')
print(x_train['text'])
print('----------')
print(x_train['text'].iloc[:])
print('----------')
print(type(x_train['text'].iloc[:]))  ## series
print('----------')
print(x_train['text'].iloc[:].values)
print('----------')
print(type(x_train['text'].iloc[:].values)) ## numpy
print('----------')
x_train['text_len'] = x_train['text'].apply(lambda x: len(x.split(' ')))
print(x_train['text_len'].describe())

print('-----------------------------向量转换----------------------------------')
# 2. TfidfVectorizer，转换词向量
tfidfvec = TfidfVectorizer(max_features=3000)   ## f1_score = 0.86
# tfidfvec = TfidfVectorizer(ngram_range=(1,3), max_features=3000)    # f1_score = 0.87
x_train_vec = tfidfvec.fit_transform(x_train['text'])
x_test_vec = tfidfvec.fit_transform(x_test['text'])

print('-----------------------------模型训练----------------------------------')
# 3. 使用模型训练(使用前一万条数据训练，后边的数据做模型评估)
clf = RidgeClassifier()
clf.fit(x_train_vec[:10000],x_train['label'].values[:10000])

print('-----------------------------模型评估----------------------------------')
# 4. 模型预测及评估
x_train_pred = clf.predict(x_train_vec[10000:])
print('f1_score = ' + str(f1_score(x_train['label'].values[10000:],x_train_pred, average='macro')))

print('-----------------------------模型预测+结果输出----------------------------------')
# 5. 结果输出
x_test_pred = clf.predict(x_test_vec)
x_test['label'] = x_test_pred
# x_test.to_csv(r"result_tf-idf_15000row.csv", index=False)  # 15000测试
x_test['label'].to_csv("result_tf-idf.csv", index=False)    # 跑全部数据集
