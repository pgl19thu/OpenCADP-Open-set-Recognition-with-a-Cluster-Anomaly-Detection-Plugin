import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from ae import AE
import random
import joblib
import os


def shuffle_tensor(tensors):
    index = [i for i in range(len(tensors))]
    random.shuffle(index)
    return tensors[index]


def anomaly_model(save_dir, data_train, ctm, all_class):
    data = data_train[:, :-1]
    label = data_train[:, -1]
    for c in all_class:
        data_class = data[label == c]
        data_class = shuffle_tensor(data_class)

        iso = IsolationForest(random_state=0, n_estimators=250, contamination=ctm).fit(data_class)
        joblib.dump(iso, save_dir + 'iso' + str(c) + '.pkl')  # 读取Model ell = joblib.load('save/ell.pkl')

        ell = EllipticEnvelope(contamination=ctm, random_state=0).fit(data_class)
        joblib.dump(ell, save_dir + 'ell' + str(c) + '.pkl')  # 读取Model ell = joblib.load('save/ell.pkl')

        ocsvm = OneClassSVM(gamma='auto', nu=ctm).fit(data_class)
        joblib.dump(ocsvm, save_dir + 'ocsvm' + str(c) + '.pkl')  # 读取Model ell = joblib.load('save/ell.pkl')

        lof = LocalOutlierFactor(contamination=ctm, novelty=True).fit(data_class)
        joblib.dump(lof, save_dir + 'lof' + str(c) + '.pkl')  # 读取Model ell = joblib.load('save/ell.pkl')

        clf = AE()
        clf.fit(data_class.astype(float), save_dir + 'ae' + str(c) + '.pkl', contamin=ctm)
        np.save(save_dir + 'ae' + str(c) + '_thresholds.npy', np.array(clf.threshold))


def get_recall(result_ell_cls, test_label, clss):
    yt = result_ell_cls[test_label == clss]
    yo = result_ell_cls[test_label != clss]
    recall_c = np.sum(yt == -1) / len(yt)
    recall_o = np.sum(yo == -1) / len(yo)
    print('testing recall of class', np.sum(yt == -1), len(yt))
    print('other recall of class', np.sum(yo == -1), len(yo))
    print('testing recall of class', clss, recall_c)
    print('other recall of class', clss, recall_o)
    return recall_c, recall_o


def anomaly_detect(save_dir, data_test, all_class, ctm):  # 输入必须为二维
    results = []
    best_results = []
    temp = np.ones((len(data_test),))
    label_test = all_class[data_test[:, -1].astype(int)]
    for clss in all_class:
        best_model = None
        best_result = temp
        best_recall_other = 0

        ell = joblib.load(save_dir + 'ell' + str(clss) + '.pkl')
        result_ell_cls = ell.predict(data_test[:, :-1])

        recall_ell, recall_other_ell = get_recall(result_ell_cls, label_test, clss)
        if recall_ell > ctm * 2 or recall_other_ell * 0.2 < recall_ell:
            result_ell_cls = temp
        else:
            best_model = ell
            best_recall_other = recall_other_ell
            best_result = result_ell_cls

        ocsvm = joblib.load(save_dir + 'ocsvm' + str(clss) + '.pkl')
        result_ocsvm_cls = ocsvm.predict(data_test[:, :-1])
        recall_ocsvm, recall_other_ocsvm = get_recall(result_ocsvm_cls, label_test, clss)
        if recall_ocsvm > ctm * 2 or recall_other_ocsvm* 0.2 < recall_ocsvm:
            result_ocsvm_cls = temp
        elif best_model is None or recall_other_ocsvm > best_recall_other:
            best_model = ocsvm
            best_recall_other = recall_other_ocsvm
            best_result = result_ocsvm_cls

        iso = joblib.load(save_dir + 'iso' + str(clss) + '.pkl')
        result_iso_cls = iso.predict(data_test[:, :-1])
        recall_iso, recall_other_iso = get_recall(result_iso_cls, label_test, clss)
        if recall_iso > ctm * 2 or recall_other_iso* 0.2 < recall_iso:
            result_iso_cls = temp
        elif best_model is None or recall_other_iso > best_recall_other:
            best_model = iso
            best_recall_other = recall_other_iso

        lof = joblib.load(save_dir + 'lof' + str(clss) + '.pkl')
        result_lof_cls = lof.predict(data_test[:, :-1])
        recall_lof, recall_other_lof = get_recall(result_lof_cls, label_test, clss)
        if recall_lof > ctm * 2 or recall_other_lof* 0.2 < recall_lof:
            result_lof_cls = temp
        elif best_model is None or recall_other_lof > best_recall_other:
            best_model = lof
            best_recall_other = recall_other_lof
            best_result = result_lof_cls

        clf = torch.load(save_dir + 'ae' + str(clss) + '.pkl')
        clf.threshold = np.load(save_dir + 'ae' + str(clss) + '_thresholds.npy')
        data = data_test[:, :-1]
        result_ae_cls, loss, hidden = clf.predict(data.astype(float))
        recall_ae, recall_other_ae = get_recall(result_ae_cls, label_test, clss)
        if recall_ae > ctm * 2 or recall_other_ae* 0.2 < recall_ae:
            result_ae_cls = temp
            if best_model is not None:
                joblib.dump(best_model, save_dir + 'best' + str(clss) + '.pkl')
        elif best_model is None or recall_other_ae > best_recall_other:
            torch.save(clf, save_dir + 'best' + str(clss) + 'ae.pkl')
            np.save(save_dir + 'best' + str(clss) + 'ae_thresholds.npy', np.array(clf.threshold))
            best_result = result_ae_cls
        else:
            joblib.dump(best_model, save_dir + 'best' + str(clss) + '.pkl')  # 读取Model ell = joblib.load('save/ell.pkl')
        result = [result_ell_cls, result_ocsvm_cls, result_iso_cls, result_lof_cls, result_ae_cls]
        results.append(result)
        best_results.append(best_result)
    np.save(save_dir + 'results.npy', np.array(results))
    np.save(save_dir + 'best_results.npy', np.array(best_results))
    print('end of ', clss)


def anomaly_predict(save_dir, data_tests, maxclasses):  # 输入必须为二维
    maxclasss = np.unique(maxclasses)

    best_clfs = []
    dict = {}
    for index, maxclass in enumerate(maxclasss):
        dict[maxclass] = index
        if os.path.exists(save_dir + 'best' + str(maxclass) + 'ae_thresholds.npy'):
            clff = torch.load(save_dir + 'best' + str(maxclass) + 'ae.pkl')
            clff.threshold = np.load(save_dir + 'best' + str(maxclass) + 'ae_thresholds.npy')
            best_clfs.append(clff)
        else:
            best_clfs.append(joblib.load(save_dir + 'best' + str(maxclass) + '.pkl'))

    best_result = []
    for index, maxclass in enumerate(maxclasses):
        if os.path.exists(save_dir + 'best' + str(maxclass) + 'ae_thresholds.npy'):
            re1, re2, re3 = best_clfs[dict[maxclass]].predict(data_tests[index:index + 1])
            best_result.append(re1)
        else:
            best_result.append(best_clfs[dict[maxclass]].predict(data_tests[index:index + 1]))
    return [best_result]


if __name__ == '__main__':
    '''设置基本参数'''
    SAVE_DATA_DIR = 'osd/ids2017_train_data.npy'
    data_train = np.load(SAVE_DATA_DIR, allow_pickle=True)
    dict_class = np.unique(data_train[:, -1])
    for i in range(10):
        SAVE_DIR = 'model/anomaly' + str(i) + '/'
        isExists = os.path.exists(SAVE_DIR)
        if not isExists:
            os.makedirs(SAVE_DIR)
        ctm = 0.001 * (i + 1)
        anomaly_model(SAVE_DIR, data_train, ctm, dict_class)
