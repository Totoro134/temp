import csv
import os
import numpy as np


# 从csv文件读取数字数据
def read_csv(csv_path):
    with open(csv_path, encoding="utf-8") as f:
        file = []
        cnt = 0
        for i in csv.reader(f):
            file.append(i)
            cnt += 1
        f.close()
    return np.array(file, dtype='float64')

def prepare_data():
    current_path = os.path.dirname(__file__)
    data = read_csv(os.path.join(current_path,'x_test.csv'))
    x1 = data[:,:3]
    x2 = data[:,3:5]
    x3 = data[:,5:]
    print(data)
    print(x1)
    print(x2)
    print(x3)
    np.savetxt('x_test1.csv', x1, delimiter=',')
    np.savetxt('x_test2.csv', x2, delimiter=',')
    np.savetxt('x_test3.csv', x3, delimiter=',')

prepare_data()