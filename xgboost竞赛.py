#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVR
from xgboost import plot_importance
from sklearn.preprocessing import Imputer
from sklearn import neighbors



def trainandTest(X_train, y_train, X_test):
    # XGBoost训练过程
    model = xgb.XGBRegressor(reg_alpha=3,reg_lambd=1,gamma = 0.1,min_child_weight=5,max_depth=6, learning_rate=0.25, n_estimators=300)
    # model = SVR()
    model.fit(X_train, y_train)

    # 对测试集进行预测
    ans = model.predict(X_test)


    # np_data = np.array([i-1 for i in ans])

    # 写入文件
    pd_data = pd.DataFrame(ans, columns=[ 'prediction'])
    # print(pd_data)
    pd_data.to_csv('/home/chase/upczxj/jingsai/competition/submitxg.csv', index=None)

    # 显示重要特征
    # plot_importance(model)
    # plt.show()

#
# def gettime(pathopen,pathclose):
#     trainfirstline = pd.read_csv(filepath_or_buffer=pathopen)
#     data_num = len(trainfirstline)
#     XList = []
#     for row in range(0, data_num):
#         tmp_list = []
#         str = trainfirstline.iloc[row][0]
#         tmp_list.append(str[11:13])
#         XList.append(tmp_list)
#     pd.DataFrame(XList).to_csv(filepath_or_buffer = pathclose)

if __name__ == '__main__':

    # trainfirstline = pd.read_csv('/home/chase/upczxj/jingsai/competition/test_4.csv')
    # data_num = len(trainfirstline)
    # XList = []
    # for row in range(0, data_num):
    #     tmp_list = []
    #     str = trainfirstline.iloc[row][1]
    #     tmp_list.append(str[11:13])
    #     XList.append(tmp_list)
    # print(XList)
    # pd.DataFrame(XList).to_csv(r'/home/chase/upczxj/jingsai/competition/t4.csv')
    # print("shuaishuaide wuwu")
    # pathopen1 = '/home/chase/upczxj/jingsai/competition/train_1.csv'
    # pathclose1 = '/home/chase/upczxj/jingsai/competition/t1.csv'
    # pathopen2 = '/home/chase/upczxj/jingsai/competition/train_2.csv'
    # pathclose2 = '/home/chase/upczxj/jingsai/competition/t2.csv'
    # pathopen3 = '/home/chase/upczxj/jingsai/competition/train_3.csv'
    # pathclose3 = '/home/chase/upczxj/jingsai/competition/t3.csv'
    # pathopen4 = '/home/chase/upczxj/jingsai/competition/train_4.csv'
    # pathclose4 = '/home/chase/upczxj/jingsai/competition/t4.csv'
    # gettime(pathopen1,pathclose1)
    # gettime(pathopen2,pathclose2)
    # gettime(pathopen3,pathclose3)
    # gettime(pathopen4,pathclose4)

    #
    # import sys
    # sys.exit()




    # 读取数据文件
    train_1 = pd.read_csv('/home/chase/upczxj/jingsai/competition/train_1.csv').values
    train_2 = pd.read_csv('/home/chase/upczxj/jingsai/competition/train_2.csv').values
    train_3 = pd.read_csv('/home/chase/upczxj/jingsai/competition/train_3.csv').values
    train_4 = pd.read_csv('/home/chase/upczxj/jingsai/competition/train_4.csv').values
    test_1 = pd.read_csv('/home/chase/upczxj/jingsai/competition/test_1.csv').values
    test_2 = pd.read_csv('/home/chase/upczxj/jingsai/competition/test_2.csv').values
    test_3 = pd.read_csv('/home/chase/upczxj/jingsai/competition/test_3.csv').values
    test_4 = pd.read_csv('/home/chase/upczxj/jingsai/competition/test_4.csv').values

    # 数据合并

    Con_train = np.vstack((train_1, train_2, train_3, train_4))
    Con_testo = np.vstack((test_1, test_2, test_3, test_4))
    Con_test = Con_testo[:, 2:9]
    Con_t = Con_train[:, 1:8]



    y = Con_train[:, 9]
    y = [i if i >= 0 else 0 for i in y]
    # y=np.array([i+1 for i in y])

    Con_data = np.vstack((Con_t, Con_test))

    # print(Con_t.shape)
    # print(Con_test.shape)
    # print(Con_data.shape)
    #
    # import sys
    # sys.exit()

    new_data = preprocessing.scale(Con_data)
    X = new_data[:183094, :]
    testdata = new_data[183094:, :]

    #
    # print(X.shape)
    # print(y.shape)
    # print(testdata.shape)
    #
    # import sys
    # sys.exit()

    trainandTest(X, y, testdata)
