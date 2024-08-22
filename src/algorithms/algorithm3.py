import torch
import numpy as np
class Algorithm3():
    def __init__(self, labelIndex, dataset):

        self.labelIndex = labelIndex
        self.dataset = dataset

    def getDataNumber(self): # 得到数量
        each_client_data_number = []
        for i in range(len(self.labelIndex)):
            each_client_data_number.append(len(self.labelIndex[i]))
        return each_client_data_number



    def getProba(self):
        each_client_data_number = self.getDataNumber()

        proba = [(each_client_data_number[i]) / sum(each_client_data_number)
                 for i in range(len(each_client_data_number))]
        return proba
