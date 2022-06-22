import numpy as np
import pywt


def timeFeature(subData):
    """
    时域特征参数
    :param subData: 波形数据，一维数据
    :return: ["平均值", "方差", "能量", "均方根", "标准差", "最大值", "最小值", "峰值系数", "峭度因子", "偏度因子", "波形系数", "脉冲因子", "裕度因子"]
    """
    subData = np.array(subData)
    # -------时域特征-------
    # 平均值
    Mean = np.mean(subData)
    # 方差
    Var = np.var(subData, ddof=1)
    # 平均幅值
    MeanAmplitude = np.mean(np.abs(subData))
    # 能量
    Energy = np.sum(np.power(subData, 2))
    # 均方根
    RMS = np.sqrt(np.mean(np.power(subData, 2)))
    # 方根幅值
    SquareRootAmplitude = np.power(np.mean(np.sqrt(np.abs(subData))), 2)
    # 标准差
    STD = np.std(subData, ddof=1)
    # 最大值
    Max = np.mean(np.sort(subData)[-10:])
    # 最小值
    Min = np.mean(np.sort(subData)[:10])

    # -------波形特征-------
    # 峰值
    Peak = np.mean(np.sort(subData)[-10:])
    # 峰值系数
    if RMS == 0:
        Cf = 0
    else:
        Cf = np.mean(np.abs(np.sort(subData)[-10:]))/RMS
    # 峭度
    Kurtosis = np.sum((subData - Mean)**4)
    # 峭度因子
    if STD == 0:
        KurtosisFactor = 0
        SkewnessFactor = 0
    else:
        KurtosisFactor = Kurtosis/((STD**4)*(len(subData - 1)))
        # 偏度因子
        SkewnessFactor = np.sum((np.abs(subData) - Mean) ** 3) / ((STD ** 3) * (len(subData - 1)))
    # 波形系数
    if Mean == 0:
        Cs = 0
    else:
        Cs = RMS/Mean
    # 脉冲因子
    if MeanAmplitude == 0:
        ImpulseFactor = 0
    else:
        ImpulseFactor = Cf/MeanAmplitude
    # 裕度因子
    if SquareRootAmplitude == 0:
        MarginFactor = 0
    else:
        MarginFactor = Cf/SquareRootAmplitude

    return [Mean, Var, Energy, RMS, STD, Max, Min, Cf, KurtosisFactor, SkewnessFactor, Cs, ImpulseFactor, MarginFactor]


def freFeature(Fre, FFT_y):
    """
    频域特征提取
    :param Fre: FFT变换后的频率
    :param FFT_y: FFT变换后的幅值
    :return: [频率幅值平均值, 重心频率, 均方根频率, 标准差频率, 频率集中度, 频率峭度]
    """
    # 频率幅值平均值
    S1 = np.sum(FFT_y)/len(FFT_y)
    # 重心频率
    S2 = np.sum(np.array(Fre)*np.array(FFT_y))/np.sum(FFT_y)
    # 均方根频率
    S3 = np.sqrt(np.sum(np.array(FFT_y)*np.array(FFT_y))/len(FFT_y))
    # 标准差频率
    S4 = np.sqrt(np.sum((np.array(FFT_y) - S1) * (np.array(FFT_y) - S1)) / len(FFT_y))
    # 频率集中度（向重心频率靠拢的集中度）
    S5 = 1-(np.sum(np.abs(np.array(FFT_y)*np.array(Fre-S2)))/np.sum(np.array(FFT_y)*np.array(Fre)))
    # 频率峭度
    S6 = np.sum((np.array(FFT_y)-S1)**4)/(len(FFT_y)*(S4**4))

    return np.array([S1, S2, S3, S4, S5, S6])


def __threshold(data, name="minimaxi", flag=False):
    # 阈值选取
    delta = 0
    if name == "minimaxi":
        delta = 0.3936 + 0.1829 * (np.log(len(data)) / np.log(2))

    if name == "sqtwolog":
        delta = np.sqrt(2 * np.log(len(data)))

    if name == "rigrsure":
        eta = (np.sum(np.square(data)) - len(data))/len(data)
        criti = np.sqrt((1/len(data)*(np.log(len(data))/np.log(2))))
        if eta < criti:
            delta = np.sqrt(2 * np.log2(len(data)))
        else:
            delta = min([np.sqrt(2 * np.log2(len(data))), np.sqrt(2 * np.log2(len(data)))])

    if name == "ure":
        f = np.square(np.sort(np.abs(data)))
        rish = []
        for i in range(len(f)):
            rish.append((len(data) - 2*i + np.sum(f[:i]) + (len(data) - i)*f[len(data) - i - 1])/len(data))
        indiex = np.argmin(rish)
        delta = np.sqrt(f[indiex])

    if flag:
        sigma = np.sort(np.abs(data))[int(len(data) / 2) + 1] / 0.6745
        delta = sigma*delta

    return delta


def thresholdDenoise(data, wavelet, maxlevel, name="minimaxi", thre="soft"):
    """
    阈值去噪
    :param data: 需要去噪的波形数据
    :param wavelet: 小波基
    :param maxlevel: 分解最大层
    :param name: 阈值名称
    :param thre: ’soft‘或者’hard‘
    :return:
    """
    wpt = pywt.WaveletPacket(data, wavelet=wavelet, maxlevel=maxlevel)
    wpt1 = pywt.WaveletPacket(None, wavelet=wavelet, maxlevel=maxlevel)
    for i in range(maxlevel, maxlevel + 1 ):
        node = wpt.get_level(i, "freq")
        for loop, sub_data in enumerate(node):
            delta = __threshold(data, name, False)
            wpt1[sub_data.path] = pywt.threshold(sub_data.data, delta, thre)
    wpt1 = wpt1.reconstruct()

    return wpt1


def wptNodeFeature(data, maxlevel, wavelet, Fs):
    """
    小波包节点特征提取
    :param data: 波形数据
    :param maxlevel: 最大分解层数
    :param wavelet: 小波基函数
    :param Fs: 采样频率
    :return: 时域特征，频域特征，名字（这三个返回值都是字典，名字是字典名称）
    """
    wpt = pywt.WaveletPacket(data, wavelet=wavelet, maxlevel=maxlevel)
    name = []
    subtimeFeature = {}
    subfreFeature = {}
    for i in range(maxlevel, maxlevel + 1):
        node = wpt.get_level(i, "freq")
        for loop, sub_data in enumerate(node):
            wpt1 = pywt.WaveletPacket(None, wavelet=wavelet, maxlevel=maxlevel)
            wpt1[sub_data.path] = wpt[sub_data.path]
            wpt1 = np.array(wpt1.reconstruct())[:len(data)]
            subtimeFeature[loop] = timeFeature(wpt1)
            fre, fft_y = FFT(Fs, wpt1)
            subfreFeature[loop] = freFeature(fre, fft_y)
            name.append(loop)

    return subtimeFeature, subfreFeature, name


def FFT(Fs, data):
    """
    FFT变换
    :param Fs: 采样频率
    :param data: 波形
    :return: 频率，频率幅值
    """
    L = len(data)  # 信号长度
    N = int(np.power(2, np.ceil(np.log2(L))))  # 下一个最近二次幂
    FFT_y = (np.fft.fft(data, N)) / L * 2  # N点FFT，但除以实际信号长度 L
    Fre = (np.arange(int(N / 2)) / N) * Fs # 频率坐标
    FFT_y = FFT_y[:int(N / 2)]  # 取一半
    return Fre, np.abs(FFT_y)