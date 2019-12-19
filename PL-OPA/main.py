import scipy.io as sio
from metric import metric
import numpy as np
import random
from sklearn import preprocessing
from sklearn import model_selection
import csv

def read_data(file_name, type=1):
    data = sio.loadmat(file_name)
    out_data = {}
    if type == 1:
        out_data = data
    elif type == 2:
        #threshold = 10000000
        temp = list(zip(data['X'], data['Y'], data['Y_P']))
        random.shuffle(temp)
        data['X'], data['Y'], data['Y_P'] = zip(*temp)
        data['X'] = np.array(data['X'])
        data['Y'] = np.array(data['Y'])
        data['Y_P'] = np.array(data['Y_P'])

        data['X'] = preprocessing.scale(data['X'])
        # data['X'] = preprocessing.normalize(data['X'])

        # n = int(data['X'].shape[0])
        # if n > threshold:
        #     n = threshold
        # part = int(n * 0.8)
        # out_data['X_train'] = data['X'][0:part, :]
        # out_data['Y_train'] = ((data['Y_P'] > 0) * 1)[0:part, :]
        # out_data['X_test'] = data['X'][part:n, :]
        # out_data['Y_test'] = ((data['Y'] > 0) * 1)[part:n, :]
        out_data = data
    return out_data

def save_data(file_name, data):
    sio.savemat(file_name, data)

def get_pos_neg(y, scores):
    inf = -1e10
    scores_1 = (y == 0) * inf + scores
    scores_0 = (y == 1) * inf + scores
    # s_pos = np.argmin(scores_1)
    s_pos = np.argmax(scores_1)
    s_neg = np.argmax(scores_0)
    return s_pos, s_neg

def run(data):
    # data = read_data(data_file, 2)
    n = data['X_train'].shape[0]
    p = data['X_train'].shape[1]
    # p is the feature number of X
    q = data['Y_train'].shape[1]
    # q is the feature number of Y
    # W = np.random.random((p + 1, q))
    W = np.zeros((p + 1, q))
    C = 1

    X_train = np.append(data['X_train'], np.ones((data['X_train'].shape[0], 1)), axis=1)

    for i in range(0, n):
        x = X_train[i]

        y = data['Y_train'][i]
        scores = W.T.dot(x)
        s_pos, s_neg = get_pos_neg(y, scores)

        l_t = 1 - scores[s_pos] + scores[s_neg]
        if l_t <= 0:
        	continue
        tau = l_t / (2.0 * x.dot(x) + 1.0 / (2 * C))
        W[:, s_pos] += tau * x
        W[:, s_neg] -= tau * x

    scores = np.append(data['X_test'], np.ones((data['X_test'].shape[0], 1)), axis=1).dot(W)
    y_pred = np.zeros_like(scores).astype(int)
    indices = np.argmax(scores, axis=1)
    # print(indices)
    for i in range(0, y_pred.shape[0]):
        y_pred[i, indices[i]] = 1
    metrics = metric(data['Y_test'], y_pred)

    # test svm-------------
    # print(data['X_train'][0], '\n', data['X_train'][1])

    # from sklearn import svm
    # labels = np.zeros_like(data['Y_train'][:, 0])
    # for i in range(0, data['Y_train'].shape[0]):
    #     for j in range(0, data['Y_train'].shape[1]):
    #         if data['Y_train'][i][j] == 1:
    #             break
    #     labels[i] = j
    # clf = Lclf = svm.LinearSVC(random_state=0, tol=1e-5).fit(data['X_train'], labels)
    # y_pred = clf.predict(data['X_test'])
    # # print(y_pred)

    # temp = np.zeros_like(data['Y_test'])
    # for i in range(0, data['Y_test'].shape[0]):
    #     temp[i][y_pred[i]] = 1
    # metrics = metric(data['Y_test'], temp)
    # test svm-------------


    # for value in metrics:
    #     print('%0.4f\t' % value, end="")
    return metrics[0]

def ten_fold_run(data_file):
    data = read_data(data_file,2)
    kf = model_selection.KFold(n_splits=10)
    result=[] 
    for train_index, test_index in kf.split(data['X']):
        newData = {"X_train": data['X'][train_index], "Y_train": data['Y_P'][train_index],
                    "X_test": data['X'][test_index], "Y_test": data['Y'][test_index]}
        result.append(run(newData))
    return result

# dataStr = ["BirdSong", "FG-NET", "lost", "MSRCv2", "Soccer Player", "Yahoo! News"]
dataStr=["BirdSong"]
# uciDataStr = ["abalone","ecoli","glass","letter","winequality-red","yeast","pendigits","vowel"]
uciDataStr=["abalone","winequality-red","yeast"]
result = []
varianceResult = []
for i, _ in enumerate(dataStr):
    result_temp = ten_fold_run("../" + dataStr[i] + ".mat")
    result.append(np.mean(result_temp))
    varianceResult.append(np.var(result_temp))
print(result)
print(varianceResult)

# uci dataset test
# csvfile = open("result.csv", "w")
# writer = csv.writer(csvfile)
# for i, _ in enumerate(uciDataStr):
#     result = []
#     for j in range(1, 29):
#         result_temp=ten_fold_run("dir/UCI_processed/"+uciDataStr[i]+"/"+str(j)+".mat")
#         # result.append(np.mean(result_temp))
#         writer.writerow(result_temp)
#     # writer.writerow(result)
# csvfile.close()

# test for single data point
# for i, _ in enumerate(uciDataStr):
#     result = np.zeros((1,7))
#     for k in range(0,1):
#         for j in range(28, 29):
#             result_temp=ten_fold_run("dir/"+uciDataStr[i]+"/"+str(j)+".mat")
#             result[k,j-22]=(np.mean(result_temp))
#     print(np.mean(result,0))
