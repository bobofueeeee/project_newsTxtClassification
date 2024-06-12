from sklearn.feature_extraction.text import  TfidfVectorizer,CountVectorizer
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score

# 1. 读取数据，数据概览
x_train = pd.read_csv('D:/wk/data/personal-NLP-learning/train_set.csv',sep='\t') # nrows=200
x_test = pd.read_csv('D:/wk/data/personal-NLP-learning/test_a.csv', sep = '\t') # nrows=200


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


# 2. 使用CountVectorizer，转换词向量
countvector = CountVectorizer(max_features=3000)
x_train_vec = countvector.fit_transform(x_train['text'])

# 3. 使用模型训练(使用前一万条数据训练，后边的数据做模型评估)
clf = RidgeClassifier()
clf.fit(x_train_vec[:10000],x_train['label'].values[:10000])

# 4. 模型预测及评估
x_train_pred = clf.predict(x_train_vec[10000:])
print('f1_score = ' + str(f1_score(x_train['label'].values[10000:],x_train_pred, average='macro')))  ## f1_score = 0.74

# 5. 结果输出

