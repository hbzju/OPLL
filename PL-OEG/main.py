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
        # data['X']=data['X'][:,1:]
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

def run_omd_l2(data):
    #data = read_data(data_file, 2)
    n = data['X_train'].shape[0]
    p = data['X_train'].shape[1]
    # p is the feature number of X
    q = data['Y_train'].shape[1]
    # q is the feature number of Y
    W = np.zeros((p + 1, q))
    Q = np.zeros_like(W)
    eta = 5e-3

    X_train = np.append(data['X_train'], np.ones((data['X_train'].shape[0], 1)), axis=1)

    for i in range(0, n):
        x = X_train[i]
        y = data['Y_train'][i]
        scores = W.T.dot(x)
        s_pos, s_neg = get_pos_neg(y, scores)

        l_t = 1 - scores[s_pos] + scores[s_neg]
        if l_t <= 0:
            continue
        Q[:, s_pos] += x
        Q[:, s_neg] -= x
        W = np.exp(eta * Q - 1)

        # without approximation ------- just test
        # pos_idx_arr = np.where(y == 1)[0]
        # neg_idx_arr = np.where(y == 0)[0]
        # flag = True
        # for pos_idx in pos_idx_arr:
        #     if flag:
        #         for neg_idx in neg_idx_arr:
        #             W_temp = W
        #             W_temp[:, pos_idx] += eta * x
        #             W_temp[:, neg_idx] -= eta * x
        #             s_pos_temp, s_neg_temp = get_pos_neg(y, x.dot(W_temp))
        #             if s_pos_temp == pos_idx and s_neg_temp == neg_idx:
        #                 W = W_temp
        #                 flag = False
        #                 break
        # if flag:
            # print("Minimum cannot be found!")
            # W[:, s_pos] += eta * x
            # W[:, s_neg] -= eta * x
        # -------- end test

    scores = np.append(data['X_test'], np.ones((data['X_test'].shape[0], 1)), axis=1).dot(W)
    y_pred = np.zeros_like(scores).astype(int)
    indices = np.argmax(scores, axis=1)
    # print(indices)
    y_pred[np.arange(y_pred.shape[0]), indices] = 1
    metrics = metric(data['Y_test'], y_pred)
    # for value in metrics:
    #     print('%0.4f\t' % value, end="")
    return metrics[0]

# def run_omd_entropic(data_file):

def ten_fold_run(data_file):
    data = read_data(data_file,2)  
    kf = model_selection.KFold(n_splits=10)
    result=[]
    for train_index, test_index in kf.split(data['X']):
        newData = {"X_train": data['X'][train_index], "Y_train": data['Y_P'][train_index],
                    "X_test": data['X'][test_index], "Y_test": data['Y'][test_index]}
        result.append(run_omd_l2(newData))
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
#         result_temp=ten_fold_run("/Users/qiangyuzhou/Desktop/experiment_partial/data/生成的数据集/UCI_processed/"+uciDataStr[i]+"/"+str(j)+".mat")
#         # result.append(np.mean(result_temp))
#         writer.writerow(result_temp)
#     # writer.writerow(result)
# csvfile.close()

# test for single data point
# for i, _ in enumerate(uciDataStr):
#     result = np.zeros((1,7))
#     for k in range(0,1):
#         for j in range(27, 29):
#             result_temp=ten_fold_run("/Users/qiangyuzhou/Desktop/experiment_partial/data/生成的数据集/UCI_processed/"+uciDataStr[i]+"/"+str(j)+".mat")
#             result[k,j-22]=(np.mean(result_temp))
#     print(np.mean(result,0))
