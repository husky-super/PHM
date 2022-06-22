import numpy as np
import tensorflow as tf
import os
tf.compat.v1.disable_eager_execution()


def BoxModel(data):
    """
    箱型图
    :param data: 正常数据
    :return: DownLimit, UpLimit，上下限阈值
    """
    # 箱型图
    temp = np.sort(data)
    # 下四分位数
    Q1 = temp[int(len(data)/4)]
    # 上四分位数
    Q3 = temp[-int(len(data)/4)]
    # 四分位距离
    IQR = Q3 - Q1
    # 上限
    UpLimit = Q3 + 1.5*IQR
    # 下限
    DownLimit = Q1 - 1.5 * IQR

    return DownLimit, UpLimit


class EquipmentAutoEncoderTrain:
    """自编码器—设备状态监测-普通时频域特征-训练模型"""
    def __init__(self, lr=0.001, batch_size=50):
        self.lr = lr
        self.batch_size = batch_size

    def weight(self, shape, mean=0.0, stddev=1.0, seed=None):
        w = tf.compat.v1.random_normal(shape=shape, mean=mean, stddev=stddev, seed=seed)

        return tf.Variable(w)

    def bias(self, shape, value=0.1):
        b = tf.compat.v1.constant(shape=shape, value=value, dtype=tf.float32)

        return tf.Variable(b)

    def callback(self, time_X, j):

        return time_X[j * self.batch_size:(j + 1) * self.batch_size, :]

    def fit(self, X, shape, epoch, seed):
        """训练自编码器"""
        # 保存文件
        if os.path.exists("./自动编码器_设备状态检测_网络参数.npy"):
            os.remove("./自动编码器_设备状态检测_网络参数.npy")
        weight = {}
        # 时间训练数据
        time_x = tf.compat.v1.placeholder(tf.float32, shape=[None, X.shape[1]])
        # 定义deep层网络
        # 时域网络-编码器
        time_weight_layer1 = self.weight(shape=shape[0], seed=seed)
        time_bias_layer1 = self.bias(shape=[shape[0][1]])
        time_result_layer1 = tf.nn.tanh(tf.matmul(time_x, time_weight_layer1) + time_bias_layer1)

        time_weight_layer2 = self.weight(shape=shape[1], seed=seed)
        time_bias_layer2 = self.bias(shape=[shape[1][1]])
        time_result_layer2 = tf.matmul(time_result_layer1, time_weight_layer2) + time_bias_layer2

        # 时域网络-解码器
        time_weight_layer3 = self.weight(shape=shape[2], seed=seed)
        time_bias_layer3 = self.bias(shape=[shape[2][1]])
        time_result_layer3 = tf.nn.tanh(tf.matmul(time_result_layer2, time_weight_layer3) + time_bias_layer3)
        # 第四层
        time_weight_layer4 = self.weight(shape=shape[3], seed=seed)
        time_bias_layer4 = self.bias(shape=[shape[3][1]])
        time_result_layer4 = tf.nn.tanh(tf.matmul(time_result_layer3, time_weight_layer4) + time_bias_layer4)

        # 定义损失函数
        loss_function = tf.reduce_mean(tf.square(time_result_layer4 - time_x)) + tf.reduce_mean(tf.square(time_result_layer2))
        # 定义优化器
        step_var = tf.compat.v1.train.AdamOptimizer(self.lr).minimize(loss_function)
        loss1 = []
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            for i in range(epoch):
                for j in range(int(len(X)//self.batch_size)):
                    time_train = self.callback(X, j)
                    sess.run(step_var, feed_dict={time_x:time_train})
                loss = sess.run(loss_function, feed_dict={time_x:X})
                loss1.append(loss)
                print(f"epoch={i}, loss={loss} lr={self.lr}")

            # 保存权重
            weight["weight1"] = sess.run([time_weight_layer1, time_bias_layer1])
            weight["weight2"] = sess.run([time_weight_layer2, time_bias_layer2])
            weight["weight3"] = sess.run([time_weight_layer3, time_bias_layer3])
            weight["weight4"] = sess.run([time_weight_layer4, time_bias_layer4])
            np.save("./自动编码器_设备状态检测_网络参数.npy", weight)


class EquipmentAutoEncoderTest:
    """自编码器—设备状态监测-普通时频域特征-测试模型"""
    def weight(self):
        w = np.load("./自动编码器_设备状态检测_网络参数.npy", allow_pickle=True)
        return w

    def predict(self, X):
        """训练自编码器"""
        weight = self.weight()
        # 时间训练数据
        time_x = tf.compat.v1.placeholder(tf.float32, shape=[None, X.shape[1]])
        # 定义deep层网络
        # 时域网络-编码器
        time_weight_layer1 = tf.Variable(weight["weight1"][0])
        time_bias_layer1 = tf.Variable(weight["weight1"][1])
        time_result_layer1 = tf.nn.tanh(tf.matmul(time_x, time_weight_layer1) + time_bias_layer1)
        # 第四层
        time_weight_layer2 = tf.Variable(weight["weight2"][0])
        time_bias_layer2 = tf.Variable(weight["weight2"][1])
        time_result_layer2 = tf.matmul(time_result_layer1, time_weight_layer2) + time_bias_layer2

        # 时域网络-解码器
        time_weight_layer3 = tf.Variable(weight["weight3"][0])
        time_bias_layer3 = tf.Variable(weight["weight3"][1])
        time_result_layer3 = tf.nn.tanh(tf.matmul(time_result_layer2, time_weight_layer3) + time_bias_layer3)
        # 第四层
        time_weight_layer4 = tf.Variable(weight["weight4"][0])
        time_bias_layer4 = tf.Variable(weight["weight4"][1])
        time_result_layer4 = tf.nn.tanh(tf.matmul(time_result_layer3, time_weight_layer4) + time_bias_layer4)

        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            result = sess.run(time_result_layer2, feed_dict={time_x:X})

        return result


class EquipmentAutoEncoderSoftTrain:
    """soft+AE自编码器"""
    def __init__(self, lr=0.01, batch_size=50):
        self.lr = lr
        self.batch_size = batch_size

    def weight(self, shape, mean=0.0, stddev=1.0, seed=None):
        w = tf.compat.v1.random_normal(shape=shape, mean=mean, stddev=stddev, seed=seed)

        return tf.Variable(w)

    def bias(self, shape, value=0.1):
        b = tf.compat.v1.constant(shape=shape, value=value, dtype=tf.float32)

        return tf.Variable(b)

    def Threshold(self, x, w, length):
        """软阈值"""
        weight = tf.identity(w)
        for i in range(length - 1):
            weight = tf.concat([weight, w], 0)

        x = tf.where(tf.abs(x) > tf.abs(weight), 0.0, x)
        x = tf.where(x > tf.abs(weight), x - tf.abs(weight), x)
        x = tf.where(x < -tf.abs(weight), x + tf.abs(weight), x)

        return x, weight

    def callback(self, x, y, time_X, j):

        return x[j * self.batch_size:(j + 1) * self.batch_size, :], y[j * self.batch_size:(j + 1) * self.batch_size, :], time_X[j * self.batch_size:(j + 1) * self.batch_size,:]

    def fit(self, fre_X, time_X, shape, epoch, seed):
        """
        :param fre_X: 频域数据
        :param time_X: 时域数据
        :param shape: 频域网络有 8层，时域有4层 例如：np.array([[int(colon), 100], [100, 50], [50, 5], [5, 1], [1, 5], [5, 50], [50, 100], [100, int(colon)], [colon1, 20], [20, 1], [1, 20], [20, colon1]])
        :param epoch: 训练次数
        :param seed: 随机数种子
        :return:
        """
        Y = np.zeros(shape=fre_X.shape)
        # 保存文件
        if os.path.exists("./soft+AE_网络参数.npy"):
            os.remove("./soft+AE_网络参数.npy")
        weight = {}
        # 训练数据y
        x = tf.compat.v1.placeholder(tf.float32, shape=[None, fre_X.shape[1]])
        y = tf.compat.v1.placeholder(tf.float32, shape=[None, fre_X.shape[1]])
        # 时间训练数据
        time_x = tf.compat.v1.placeholder(tf.float32, shape=[None, time_X.shape[1]])
        # 定义deep层网络
        # 第一层
        weight_layer1 = self.weight(shape=shape[0], seed=seed)
        bias_layer1 = self.bias(shape=[shape[0][1]])
        weight_layer_thres1 = tf.Variable(tf.random.truncated_normal(shape=[1, fre_X.shape[1]], seed=seed))
        x1, weight_Thre = self.Threshold(x, weight_layer_thres1, self.batch_size)
        result_layer1 = tf.nn.tanh(tf.matmul(x1, weight_layer1) + bias_layer1)
        # 第二层
        weight_layer2 = self.weight(shape=shape[1], seed=seed)
        bias_layer2 = self.bias(shape=[shape[1][1]])
        result_layer2 = tf.nn.tanh(tf.matmul(result_layer1, weight_layer2) + bias_layer2)
        # 第三层
        weight_layer3 = self.weight(shape=shape[2], seed=seed)
        bias_layer3 = self.bias(shape=[shape[2][1]])
        result_layer3 = tf.nn.tanh(tf.matmul(result_layer2, weight_layer3) + bias_layer3)
        # 第四层
        weight_layer4 = self.weight(shape=shape[3], seed=seed)
        bias_layer4 = self.bias(shape=[shape[3][1]])
        result_layer4 = tf.matmul(result_layer3, weight_layer4) + bias_layer4

        # 时域网络-编码器
        time_weight_layer1 = self.weight(shape=shape[8], seed=seed)
        time_bias_layer1 = self.bias(shape=[shape[8][1]])
        time_result_layer1 = tf.nn.tanh(tf.matmul(time_x, time_weight_layer1) + time_bias_layer1)

        time_weight_layer2 = self.weight(shape=shape[9], seed=seed)
        time_bias_layer2 = self.bias(shape=[shape[9][1]])
        time_result_layer2 = tf.matmul(time_result_layer1, time_weight_layer2) + time_bias_layer2

        # 中间层
        center_weight_layer = self.weight(shape=[2, 1], seed=seed)
        center_bias_layer = self.bias(shape=[1])
        center_result = tf.matmul(tf.concat((time_result_layer2, result_layer4), axis=1), center_weight_layer) + center_bias_layer

        # 第五层
        weight_layer5 = self.weight(shape=shape[4], seed=seed)
        bias_layer5 = self.bias(shape=[shape[4][1]])
        result_layer5 = tf.nn.tanh(tf.matmul(center_result, weight_layer5) + bias_layer5)
        # 第六层
        weight_layer6 = self.weight(shape=shape[5], seed=seed)
        bias_layer6 = self.bias(shape=[shape[5][1]])
        result_layer6 = tf.nn.tanh(tf.matmul(result_layer5, weight_layer6) + bias_layer6)
        # 第七层
        weight_layer7 = self.weight(shape=shape[6], seed=seed)
        bias_layer7 = self.bias(shape=[shape[6][1]])
        result_layer7 = tf.nn.tanh(tf.matmul(result_layer6, weight_layer7) + bias_layer7)
        # 第八层
        weight_layer8 = self.weight(shape=shape[7], seed=seed)
        bias_layer8 = self.bias(shape=[shape[7][1]])
        result_layer8 = tf.matmul(result_layer7, weight_layer8) + bias_layer8

        # 时域网络-解码器
        time_weight_layer3 = self.weight(shape=shape[10], seed=seed)
        time_bias_layer3 = self.bias(shape=[shape[10][1]])
        time_result_layer3 = tf.nn.tanh(tf.matmul(center_result, time_weight_layer3) + time_bias_layer3)
        # 第四层
        time_weight_layer4 = self.weight(shape=shape[11], seed=seed)
        time_bias_layer4 = self.bias(shape=[shape[11][1]])
        time_result_layer4 = tf.nn.tanh(tf.matmul(time_result_layer3, time_weight_layer4) + time_bias_layer4)

        # 定义损失函数
        loss_function = tf.reduce_mean(tf.square(result_layer8 - y)) + tf.reduce_mean(tf.square(x - weight_Thre)) + tf.reduce_mean(tf.square(time_result_layer4 - time_x))
        # 定义优化器
        step_var = tf.compat.v1.train.AdamOptimizer(self.lr).minimize(loss_function)
        loss1 = []
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            for i in range(epoch):
                j=0
                for j in range(int(len(fre_X)//self.batch_size)):
                    x_train, y_train, time_train = self.callback(fre_X, Y, time_X, j)
                    sess.run(step_var, feed_dict={x: x_train, y: y_train, time_x:time_train})
                x_train, y_train, time_train = self.callback(fre_X, Y, time_X, j)
                loss = sess.run(loss_function, feed_dict={x: x_train, y: y_train, time_x:time_train})
                loss1.append(loss)
                print(f"epoch={i}, loss={loss} lr={self.lr}")

            # 保存权重
            weight["weight1"] = sess.run([weight_layer1, bias_layer1])
            weight["weight2"] = sess.run([weight_layer2, bias_layer2])
            weight["weight_layer_thres1"] = sess.run([weight_layer_thres1])
            weight["weight3"] = sess.run([weight_layer3, bias_layer3])
            weight["weight4"] = sess.run([weight_layer4, bias_layer4])
            weight["weight5"] = sess.run([weight_layer5, bias_layer5])
            weight["weight6"] = sess.run([weight_layer6, bias_layer6])
            weight["weight7"] = sess.run([weight_layer7, bias_layer7])
            weight["weight8"] = sess.run([weight_layer8, bias_layer8])
            weight["weight9"] = sess.run([time_weight_layer1, time_bias_layer1])
            weight["weight10"] = sess.run([time_weight_layer2, time_bias_layer2])
            weight["weight11"] = sess.run([center_weight_layer, center_bias_layer])
            weight["weight12"] = sess.run([time_weight_layer3, time_bias_layer3])
            weight["weight13"] = sess.run([time_weight_layer4, time_bias_layer4])
            np.save("./soft+AE_网络参数.npy", weight)


class EquipmentAutoEncoderSoftTest:
    """soft+AE测试模型"""
    def Threshold(self, x, w, length):
        """软阈值"""
        weight = tf.identity(w)
        for i in range(length - 1):
            weight = tf.concat([weight, w], 0)

        x = tf.where(tf.abs(x) > tf.abs(weight), 0.0, x)
        x = tf.where(x > tf.abs(weight), x - tf.abs(weight), x)
        x = tf.where(x < -tf.abs(weight), x + tf.abs(weight), x)

        return x

    def loadweight(self, io):
        """加载神经网络权重"""
        weight = np.load(io, allow_pickle=True)

        return weight

    def predict(self, fre_X, time_X):
        """
        :param fre_X: 时域数据
        :param time_X: 频域数据
        :return: 监测结果
        """
        weight = self.loadweight("./soft+AE_网络参数.npy")
        # 训练数据y
        x = tf.compat.v1.placeholder(tf.float32, shape=[None, fre_X.shape[1]])
        time_x = tf.compat.v1.placeholder(tf.float32, shape=[None, time_X.shape[1]])
        # 定义deep层网络
        # 第一层
        weight_layer1 = tf.Variable(weight["weight1"][0])
        bias_layer1 = tf.Variable(weight["weight1"][1])
        weight_layer_thres1 = tf.Variable(weight["weight_layer_thres1"][0])
        x1 = self.Threshold(x, weight_layer_thres1, fre_X.shape[0])
        result_layer1 = tf.nn.tanh(tf.matmul(x1, weight_layer1) + bias_layer1)
        # 第二层
        weight_layer2 = tf.Variable(weight["weight2"][0])
        bias_layer2 = tf.Variable(weight["weight2"][1])
        result_layer2 = tf.nn.tanh(tf.matmul(result_layer1, weight_layer2) + bias_layer2)
        # 第三层
        weight_layer3 = tf.Variable(weight["weight3"][0])
        bias_layer3 = tf.Variable(weight["weight3"][1])
        result_layer3 = tf.nn.tanh(tf.matmul(result_layer2, weight_layer3) + bias_layer3)
        # 第四层
        weight_layer4 = tf.Variable(weight["weight4"][0])
        bias_layer4 = tf.Variable(weight["weight4"][1])
        result_layer4 = tf.matmul(result_layer3, weight_layer4) + bias_layer4

        # 时域网络-编码器
        time_weight_layer1 = tf.Variable(weight["weight9"][0])
        time_bias_layer1 = tf.Variable(weight["weight9"][1])
        time_result_layer1 = tf.nn.tanh(tf.matmul(time_x, time_weight_layer1) + time_bias_layer1)
        # 第四层
        time_weight_layer2 = tf.Variable(weight["weight10"][0])
        time_bias_layer2 = tf.Variable(weight["weight10"][1])
        time_result_layer2 = tf.matmul(time_result_layer1, time_weight_layer2) + time_bias_layer2

        # 中间层
        center_weight_layer = tf.Variable(weight["weight11"][0])
        center_bias_layer = tf.Variable(weight["weight11"][1])
        center_result = tf.matmul(tf.concat((time_result_layer2, result_layer4), axis=1), center_weight_layer) + center_bias_layer

        result = {}
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            result = sess.run(center_result, feed_dict={x: fre_X, time_x:time_X})

        return result


