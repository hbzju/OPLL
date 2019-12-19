import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score

def metric(y, y_pred):
    # n = y.shape[0]
    acc = accuracy_score(y, y_pred)
    prec_mic = precision_score(y, y_pred, average='micro')
    rec_mic = recall_score(y, y_pred, average='micro')
    f1_mic = f1_score(y, y_pred, average="micro")
    prec_mac = precision_score(y, y_pred, average='macro')
    rec_mac = recall_score(y, y_pred, average='macro')
    f1_mac = f1_score(y, y_pred, average="macro")
    return acc, prec_mic, rec_mic, f1_mic, prec_mac, rec_mac, f1_mac