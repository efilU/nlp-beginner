import numpy as np  # 用于数值计算
import pandas as pd  # 用于数据处理和分析
from sklearn.model_selection import train_test_split  # 用于将数据集拆分为训练集和验证集
from sklearn.feature_extraction.text import CountVectorizer  # 用于将文本数据转换为特征向量
from sklearn.preprocessing import OneHotEncoder  # 用于将标签转换为one-hot编码
import time  # 用于时间计算
from tqdm import tqdm  # 用于过程形象化


# 读取训练集和测试集
train_df = pd.read_csv('train.tsv', sep='\t')  # tsv文件单元以'\t'作为分割
test_df = pd.read_csv('test.tsv', sep='\t')

# 检查并处理NaN值
train_df.dropna(subset=['Phrase'], inplace=True)
# test_df.dropna(subset=['Phrase'], inplace=True)

# 提取训练集中的文本和标签并转化为列表
texts_train = train_df['Phrase'].tolist()  # 选择 DataFrame 中名为 'Phrase' 的列
labels_train = train_df['Sentiment'].tolist()

# 提取测试集中的文本和对应PhraseId
texts_test = test_df['Phrase'].tolist()
phrase_ids = test_df['PhraseId'].tolist()

# 使用CountVectorizer进行特征提取
vectorizer = CountVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2))
                           # stop_words参数用于指定要去除的常见单词（停用词），通常为频率很高但对文本分类贡献不大的词汇
                           # 设置为'english'表示使用内置的英文停用词列表
                           # max_features参数用于限制特征的数量，选取词频较高的前5k词
train_x = vectorizer.fit_transform(texts_train).toarray()  # 对训练集文本进行特征提取！！！！！！！！！！！！！！写博客！！！！！！！！！！！！！！！！！！！
train_y = np.array(labels_train)

# 将标签转化为one-hot编码
encoder = OneHotEncoder(sparse=False)  # 创建一个OneHotEncoder对象，并设置返回密集数组，方便后续操作
onehot_train_y = encoder.fit_transform(train_y.reshape(-1, 1))  # 将一维标签数组转换为二维one-hot编码数组

# 拆分出验证集
train_x, val_x, train_y, val_y = train_test_split(train_x, onehot_train_y, test_size=0.3, random_state=2025)
                          # test_size表示验证集比例，random_state表示随机种子，确保每次结果一致

# 定义softmax回归模型
class SoftmaxRegression:
    def __init__(self, lr=0.01, epochs=5, batch_size=16):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.bias = None
        self.weights = None
        self.best_weights = None
        self.best_bias = None
        self.best_val_accuracy = 0.0

    # 计算softmax函数
    def softmax(self, z):  # z为对应的向量
        exp_z = np.exp(z - np.max(z))  # 减去最大值防止溢出
        return exp_z / exp_z.sum(axis=1, keepdims=True)  # 归一化处理，axis=1表示按行求和，keepdims=True表示维持原始数据的维度结构,返回为二维数组

    # 进行模型训练
    def fit(self, train_x, train_y, val_x, val_y):
        n, m = train_x.shape  # 样本数量、特征数量
        k = train_y.shape[1]  # 类别数量
        self.weights = np.zeros((m, k))  # 初始化权重矩阵，即每个特征对应每个类别的权重
        self.bias = np.zeros(k)  # 初始化偏置向量，即每个类别的偏置

        # 开始训练
        for epoch in tqdm(range(self.epochs)):
            # 进行数据打乱
            indices = np.arange(n)  # 创建一个从0到n-1的索引数组
            np.random.shuffle(indices)  # 随机打乱，确保每次迭代时数据顺序不同
            X_shuffled = train_x[indices]    # 根据打乱后的索引重新排列x,y矩阵
            Y_shuffled = train_y[indices]
            start_time = time.time()  # 记录本轮开始时间

            # 进行分批次计算
            for start_idx in range(0, n, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n)
                X_batch = X_shuffled[start_idx:end_idx]
                Y_batch = Y_shuffled[start_idx:end_idx]

                # 前向传播
                linear_model = np.dot(X_batch, self.weights) + self.bias
                            # 计算线性组合Z=XW+b，其中X是特征矩阵，W是权重矩阵,b是偏置向量，np.dot为点乘
                Y_predicted = self.softmax(linear_model)  # 使用softmax将线性模型的输出转化为概率分布

                # 计算梯度
                dw = (1 / self.batch_size) * np.dot(X_batch.T, Y_predicted - Y_batch)
                            # np.dot表示计算特征与误差的点积，得到梯度的权重，然后再对梯度进行平均不受批量大小的影响
                db = (1 / self.batch_size) * np.sum(Y_predicted - Y_batch, axis=0)
                            # 计算偏置的梯度，沿着样本方向求和，得到每个类别的偏置梯度，再求平均

                # 更新参数，采用梯度下降法
                self.weights -= self.lr * dw
                self.bias -= self.lr * db

            # 验证集的准确率
            val_pred = self.predict(val_x)
            val_accuracy = np.mean((val_pred == np.argmax(val_y, axis=1)).astype(int))  # 计算准确率
                                    # np.argmax表示val_y中每个轴上最大值的索引，axis=1表示沿列的方向寻找，即按样本顺序
            if val_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = val_accuracy
                self.best_weights = self.weights.copy()
                self.best_bias = self.bias.copy()

            val_loss = -np.mean(np.log(np.sum(val_y * self.softmax(np.dot(val_x, self.weights) + self.bias), axis=1)))
            print("[%03d/%03d] %2.2f sec(s) valLoss: %.6f | valAcc: %.6f" % \
                  (epoch, self.epochs, time.time() - start_time, val_loss, val_accuracy))  # 打印本轮训练结果

    # 预测类别
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.softmax(linear_model)
        y_pred_cls = np.argmax(y_pred, axis=1)  # 第一维预测概率最大的作为类别
        return y_pred_cls

    # 加载最好模型参数
    def load_best_model(self):
        if self.best_weights is not None and self.best_bias is not None:
            self.weights = self.best_weights
            self.bias = self.best_bias
        else:
            raise ValueError("No best model found yet")  # 抛出异常


# 初始化Softmax回归模型
model_softmax = SoftmaxRegression(lr=0.01, epochs=5, batch_size=16)

# 训练模型
model_softmax.fit(train_x, train_y, val_x, val_y)

# 加载最好模型
model_softmax.load_best_model()

# 处理测试集中的NaN值
nan_indices = test_df[test_df['Phrase'].isnull()].index.tolist()
valid_indices = [i for i in range(len(texts_test)) if i not in nan_indices]

# 提取有效测试集特征
valid_texts_test = [texts_test[i] for i in valid_indices]
valid_phrase_ids = [phrase_ids[i] for i in valid_indices]

# 对有效特征进行测试
if valid_texts_test:
    valid_X = vectorizer.transform(valid_texts_test).toarray()  # 相较于fit_transform，这里仅转换而不改变现有词汇表
    pred_valid_test = model_softmax.predict(valid_X)
else:
    pred_valid_test = []

# 为无效样本分配默认情感标签（2）
default_sentiment = 2

# 合并预测结果
final_phrase_ids = []
final_pred_test = []

cur_valid = 0
for idx in range(len(texts_test)):
    final_phrase_ids.append(phrase_ids[idx])
    if idx in nan_indices:
        final_pred_test.append(default_sentiment)
    else:
        final_pred_test.append(pred_valid_test[cur_valid])
        cur_valid += 1

# 准备答案数据
output_data = pd.DataFrame({
    'PhraseId': final_phrase_ids,
    'Sentiment': final_pred_test
})

# 写入ans.csv文件
output_data.to_csv('ans.csv', index=False)
print("结果已写入 ans.csv，共%d行" % len(output_data))