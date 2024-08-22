import torch
import numpy as np
class Algorithm2():
    def __init__(self, localE, uploadE, localD, uploadD, labelIndex, dataset):
        self.localE = localE
        self.uploadE = uploadE
        self.localD =  localD
        self.uploadD = uploadD
        self.labelIndex = labelIndex
        self.dataset = dataset
        self.aplha = 0.5
        self.beta = 0.5
        self.omega = (0.2, 0.4, 0.4)

    def getStat(self, train_data):
        '''
        Compute mean and variance for training data
        :param train_data: 自定义类Dataset(或ImageFolder即可)
        :return: (mean, std)
        '''

        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=1, shuffle=False, num_workers=0,
            pin_memory=True)
        mean = torch.zeros(1)
        std = torch.zeros(1)
        for X in train_loader:
            for d in range(1):
                mean[d] += X[:, d, :, :].mean()
                std[d] += X[:, d, :, :].std()
        mean.div_(len(train_data))
        std.div_(len(train_data))
        return list(mean.numpy()), list(std.numpy())

    def getDataDistribution(self): # 得到分布
        each_client_data_distribution = []
        each_client_data_number = self.getDataNumber()
        # print(each_client_data_number)
        for i in range(len(self.localE)):
            a, b = self.getStat(self.dataset.trainData[self.labelIndex[i]]) # 均值和方差
            each_client_data_distribution.append(a[0])
        temp = [(each_client_data_number[i] * each_client_data_distribution[i]) for i in range(len(self.localE))]
        sum_data_distribution = sum(temp) / sum(each_client_data_number)
        result = [1 / (abs(x - sum_data_distribution) + 1) for x in each_client_data_distribution]
        each_client_data_distribution_nor = [result[i] / max(result) for i in range(len(result))]
        # each_client_data_distribution_nor = np.array(each_client_data_distribution_nor) * (1 - 2 * np.finfo(float).eps) + np.finfo(float).eps
        return each_client_data_distribution_nor

    def getDataNumber(self): # 得到数量
        each_client_data_number = []
        label_own = [[] for _ in range(len(self.localE))] # 每个设备拥有哪些标签
        for i in range(len(label_own)):
            label_own[i] = self.dataset.trainLabel[self.labelIndex[i]]
            each_client_data_number.append(len(label_own[i]))
        # each_client_data_number_nor = [each_client_data_number[i] / max(each_client_data_number) for i in range(len(each_client_data_number))]

        # 开区间
        # each_client_data_number_nor = np.array(each_client_data_number_nor) * (1 - 2 * np.finfo(float).eps) + np.finfo(float).eps

        return each_client_data_number


    def getDataIMB(self): # 基尼系数
        label_own = [[] for _ in range(len(self.localE))] # 每个设备拥有哪些标签
        each_label_own_number = [[] for _ in range(len(self.localE))] #
        D_imb = [[] for _ in range(len(self.localE))]
        for i in range(len(label_own)):
            label_own[i] = self.dataset.trainLabel[self.labelIndex[i]]
            for j in range(10):
                each_label_own_number[i].append(label_own[i].tolist().count(j))
            each_label_own_number[i] = [each_label_own_number[i][j] / sum(each_label_own_number[i]) for j in range(len(each_label_own_number[i]))]

        for i in range(len(each_label_own_number)):
            D_imb[i] = 1 - sum([each_label_own_number[i][j] ** 2 for j in range(len(each_label_own_number[i]))])

        return D_imb

    def dataQuality(self):

        return
    def getProba(self):
        C_SCORE = [1 / (self.aplha * self.localD[i] / max(self.localD) + (1 - self.aplha) *
                        self.localE[i] / max(self.localE)) for i in range(len(self.localE))]

        B_SCORE = [1 / (self.aplha * self.uploadD[i] / max(self.uploadD) + (1 - self.aplha) *
                        self.uploadE[i] / max(self.uploadE)) for i in range(len(self.uploadD))]
        D_imb = self.getDataIMB()
        D_number = self.getDataNumber()
        D_number = [D_number[i] / max(D_number) for i in range(len(D_number))]
        D_dis = self.getDataDistribution()
        D_SCORE = [D_dis[i] * D_number[i] * D_imb[i]  for i in range(len(self.localE))]


        D_SCORE_N = [D_SCORE[i] / sum(D_SCORE) for i in range(len(D_SCORE))]
        D_SCORE_N = np.array(D_SCORE_N) * (1 - 2 * np.finfo(float).eps) + np.finfo(float).eps
        C_SCORE_N = [C_SCORE[i] / sum(C_SCORE) for i in range(len(C_SCORE))]
        C_SCORE_N = np.array(C_SCORE_N) * (1 - 2 * np.finfo(float).eps) + np.finfo(float).eps
        B_SCORE_N = [B_SCORE[i] / sum(B_SCORE) for i in range(len(B_SCORE))]
        B_SCORE_N  = np.array(B_SCORE_N) * (1 - 2 * np.finfo(float).eps) + np.finfo(float).eps


        sum1 = 3 * sum(C_SCORE_N) + 5 * sum(B_SCORE_N)

        proba = [(3 * C_SCORE_N[i] + 5 * B_SCORE_N[i]) / sum1  for i in range(len(C_SCORE_N))]
        # print(sum(proba))
        # print(self.clientGroup['client1'])
        # print(proba)
        return proba
