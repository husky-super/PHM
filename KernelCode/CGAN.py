import tensorflow as tf
import numpy as np
from sklearn.preprocessing import OneHotEncoder
tf.compat.v1.disable_eager_execution()
tf.compat.v1.reset_default_graph()


class CGANTrain:
    """
    CGAN训练模型
    """
    def __init__(self, g_shape, d_shape, lr=0.001, nbanch=10, epoch=10000, n_dim=3, seed=(10, 10, 10, 10)):
        """
        :param g_shape: 生成器隐藏层神经元个数，格式：(number1, number2, number3),一共三个隐藏层，最后一层为输入数据维度
        :param d_shape: 判别器隐藏层神经元个数，格式：(number1, number2),一共两个隐藏层
        :param lr: 学习率
        :param nbanch: 每次训练送入数据的个数
        :param epoch: 训练次数
        :param n_dim: 产生噪声的维度
        :param seed: 随机数种子，确定的随机数种子可以使网络波动较小
        """
        self.lr = lr
        self.nbanch = nbanch
        self.epoch = epoch
        self.seed = seed
        self.g_shape = g_shape
        self.d_shape = d_shape
        self.n_dim = n_dim

    def __ganNoise(self, shape, var):
        """
        产生噪声
        :param shape: 噪声的维度
        :param var: 方差
        :return:
        """
        # 均值和方差
        self.mean = tf.Variable(tf.compat.v1.random.normal(shape=shape, seed=self.seed[0]))
        self.var = tf.Variable(tf.compat.v1.random.normal(shape=shape, seed=self.seed[1]))
        # 产生噪音
        seed = tf.random.normal(shape=shape, stddev=var)
        self.noise = seed * self.var + self.mean
        for i in range(self.nbanch - 1):
            seed = tf.random.normal(shape=shape)
            noise = seed * self.var + self.mean
            self.noise = tf.concat([self.noise, noise], 0)

    def __weights(self, shape):
        # 权重
        return tf.compat.v1.Variable(tf.compat.v1.random.normal(shape=shape, seed=self.seed[2]))

    def __baiss(self, shape):
        # 权重
        return tf.compat.v1.Variable(tf.compat.v1.random.normal(shape=shape, seed=self.seed[3]))

    def __generator(self, noise, time):
        # 生成器
        # noise与time合成
        noise = tf.compat.v1.concat([noise, time], 1)
        # 第一层
        self.w1 = self.__weights([self.n_dim+1, self.g_shape[0]])
        self.b1 = self.__baiss([1, self.g_shape[0]])
        self.re1 = tf.tanh(tf.matmul(noise, self.w1) + self.b1)
        # 第二层
        self.w2 = self.__weights([self.g_shape[0], self.g_shape[1]])
        self.b2 = self.__baiss([1, self.g_shape[1]])
        self.re2 = tf.tanh(tf.matmul(self.re1, self.w2) + self.b2)
        # 第三层
        self.w3 = self.__weights([self.g_shape[1], self.g_shape[2]])
        self.b3 = self.__baiss([1, self.g_shape[2]])
        self.generSample = tf.tanh(tf.matmul(self.re2, self.w3) + self.b3)

    def __discriminator(self, x):
        # 判别器
        # 第一层
        self.w11 = self.__weights([self.g_shape[2], self.d_shape[0]])
        self.b11 = self.__baiss([1, self.d_shape[0]])
        self.re11 = tf.tanh(tf.matmul(x, self.w11) + self.b11)
        # 第二层
        self.w21 = self.__weights([self.d_shape[0], self.d_shape[1]])
        self.b21 = self.__baiss([1, self.d_shape[1]])
        self.re21 = tf.tanh(tf.matmul(self.re11, self.w21) + self.b21)
        # 第三层
        self.w31 = self.__weights([self.d_shape[1], 2])
        self.b31 = self.__baiss([1, 2])
        self.y = tf.compat.v1.nn.softmax(tf.matmul(self.re21, self.w31) + self.b31)

    def __makelabel(self, X):
        Xlable = np.ones([X.shape[0], 1])
        labelNoise = np.zeros([self.nbanch, 1])
        label = np.vstack((Xlable, labelNoise))
        ohe = OneHotEncoder()
        label = ohe.fit_transform(label).toarray()
        Xlabel = label[:len(Xlable), :]
        labelNoise = label[len(Xlable):, :]
        time = np.array(np.linspace(0, 1, X.shape[0]), dtype=np.float).reshape([-1, 1])

        return Xlabel, labelNoise, time

    def fit(self, X, var1):
        """
        训练模型，并保存训练完成的模型
        :param X: 训练数据
        :param var1: 方差
        :return:
        """
        # 制作标签
        Xlabel, labelNoise, time = self.__makelabel(X)
        # 占位符
        x = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, X.shape[1]])
        x1 = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, X.shape[1]])
        label = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, Xlabel.shape[1]])
        time1 = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, time.shape[1]])
        # 产生噪声
        self.__ganNoise([1, self.n_dim], var1)
        # 生成器
        self.__generator(self.noise, time1)
        # 判别器
        self.__discriminator(x)
        # 损失函数
        loss_func = tf.reduce_mean(tf.square(self.generSample - x1)) + tf.reduce_mean(tf.abs(label - self.y)) +\
                    tf.reduce_mean(tf.abs(self.var - var1)) + tf.reduce_mean(tf.abs(self.mean))
        # 梯度下降方法
        grand = tf.compat.v1.train.AdamOptimizer(self.lr).minimize(loss_func)
        # 准确率
        acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.y), tf.argmax(label)), tf.float32))

        loss_re = []
        acc_re = []
        weight = {}
        # 运行
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            for epoch in range(self.epoch):
                for banch in range(len(X)//self.nbanch):
                    subtime = time[banch * self.nbanch:(banch + 1) * self.nbanch, :]
                    x_noise = sess.run(self.generSample, feed_dict={time1:subtime})
                    x_train = X[banch*self.nbanch:(banch+1)*self.nbanch, :]
                    train = np.vstack((x_train, x_noise))
                    y_train = Xlabel[banch*self.nbanch:(banch+1)*self.nbanch, :]
                    y = np.vstack((y_train, labelNoise))
                    sess.run(grand, feed_dict={x:train, label:y, x1:x_train, time1:subtime})
                loss = sess.run(loss_func, feed_dict={x:train, label:y, x1:x_train, time1:subtime})
                loss_re.append(loss)
                acc1 = sess.run(acc, feed_dict={x:train, label:y, x1:x_train, time1:subtime})
                acc_re.append(acc1)
                print(f"loss={loss}, epoch={epoch}, acc={acc1}")

            weight["w1"] = sess.run([self.w1, self.b1])
            weight["w2"] = sess.run([self.w2, self.b2])
            weight["w3"] = sess.run([self.w3, self.b3])
            np.save("CGAN模型参数_dict.npy", weight, allow_pickle=True)


class CGANGenerator:
    """
    CGAN生成模型
    """
    def __init__(self, n_dim=3, seed=(10, 10)):
        """
        :param n_dim: 产生噪声的维度
        :param seed: 随机数种子
        """
        self.seed = seed
        self.n_dim = n_dim
        self.__loadWeight("./CGAN模型参数_dict.npy")

    def __loadWeight(self, io):
        """
        加载模型权重
        :param io: 路径
        :return:
        """
        self.weight = np.load(io, allow_pickle=True).item()

    def __ganNoise(self, shape, var):
        # 均值和方差
        self.mean = tf.Variable(tf.compat.v1.random.normal(shape=shape, seed=self.seed[0]))
        self.var = tf.Variable(tf.compat.v1.random.normal(shape=shape, seed=self.seed[1]))
        # 产生噪音
        seed = tf.random.normal(shape=shape, stddev=var)
        self.noise = seed * self.var + self.mean

    def __generator(self, noise, time):
        # 生成器
        # noise与time合成
        noise = tf.compat.v1.concat([noise, time], 1)
        # 第一层
        self.w1 = tf.Variable(self.weight["w1"][0])
        self.b1 = tf.Variable(self.weight["w1"][1])
        self.re1 = tf.tanh(tf.matmul(noise, self.w1) + self.b1)
        # 第二层
        self.w2 = tf.Variable(self.weight["w2"][0])
        self.b2 = tf.Variable(self.weight["w2"][1])
        self.re2 = tf.tanh(tf.matmul(self.re1, self.w2) + self.b2)
        # 第三层
        self.w3 = tf.Variable(self.weight["w3"][0])
        self.b3 = tf.Variable(self.weight["w3"][1])
        self.generSample = tf.tanh(tf.matmul(self.re2, self.w3) + self.b3)

    def __makelabel(self, number):
        time = np.array(np.linspace(0, 1, number), dtype=np.float).reshape([-1, 1])

        return time

    def predict(self, var1, number):
        """
        产生数据
        :param var1: 方差
        :param number: 生成数据数量
        :return:
        """
        # 产生时间序列
        time = self.__makelabel(number)
        time1 = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, time.shape[1]])
        # 产生噪声
        self.__ganNoise([1, self.n_dim], var1)
        # 生成器
        self.__generator(self.noise, time1)

        newSampleList = []
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            for i in range(len(time)):
                newSample = sess.run(self.generSample, feed_dict={time1:[time[i, :]]})
                newSampleList.append(newSample[0])

        return newSampleList


def KL(data1, data2, n=10):
    """
    计算KL散度
    :param data1: 样本1
    :param data2: 样本2
    :param n: 样本分割份数
    :return: KL散度
    """
    # 计算KL分布
    KL_max = []
    KL_min = []
    for i in range(data1.shape[1]):
        KL_max.append(max([max(data1[:, i]), max(data2[:, i])]))
        KL_min.append(min([min(data1[:, i]), min(data2[:, i])]))
    # 计算KL
    data = []
    P1 = []
    P2 = []
    for i in range(data1.shape[1]):
           P1.append(__calP(np.sort(data1[:, i]), KL_max[i], KL_min[i], n, (KL_max[i] - KL_min[i])/n))
           P2.append(__calP(np.sort(data2[:, i]), KL_max[i], KL_min[i], n, (KL_max[i] - KL_min[i])/n))

    KL = 0
    for i in range(len(P1)):
        for j in range(len(P1[0])):
            if P1[i][j] == 0 or P2[i][j] == 0:
                KL += 0
            else:
                KL += P1[i][j]*np.log(P1[i][j]/(P2[i][j]))

    return KL/len(P1)


def __calP(data, max, min, n, num):
    """
    计算数据概率分布
    :param data: 数据样本
    :param max: 最大值
    :param min: 最小值
    :param n: 样本分割份数
    :param num: 每一份样本值
    :return: 概率分布
    """
    interval = []
    P = [0]*n
    for i in range(n):
        if i == n - 1:
            interval.append([min + i * num, max])
        else:
            interval.append([min+i*num, min+(i+1)*num])

    for i in data:
        for index, j in enumerate(interval):
            if i < j[1] and i >= j[0]:
                P[index] += 1

    P = np.array(P)/len(data)

    return P
