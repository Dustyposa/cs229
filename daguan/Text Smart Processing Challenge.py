import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer


print("start calculation ...")

df_train = pd.read_csv("./train_set.csv")  # 读取csv
df_test = pd.read_csv("./test_set.csv")  # 读取csv
df_train.drop(columns=["article", "id"], inplace=True)  # 丢弃特征
df_test.drop(columns=["article"], inplace=True)  # 丢弃特征

vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, max_features=100000)
vectorizer.fit(df_train["word_seg"])  # 文本特征的提取，
x_train = vectorizer.transform(df_train["word_seg"])  # 矩阵化
x_test = vectorizer.transform(df_test["word_seg"])  # 矩阵化
y_train = df_train["class"] - 1

lg = LogisticRegression(C=4, dual=True)  # 逻辑回归训练器初始化
lg.fit(x_train, y_train)  # 训练
y_test = lg.predict(x_test)  # 预测

df_test["class"] = y_test.tolist()  # 转成list
df_test["class"] = df_test["class"] + 1  # 字段赋值
df_result = df_test.loc[:, ["id", "class"]]
df_result.to_csv("./result.csv", index=False)  # 写入csv

print("end......")