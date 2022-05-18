import numpy as np
import torch
import evt_fitting
from sklearn import svm
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler
import libmr
from sklearn.svm import OneClassSVM
import joblib
from function import shuffle_tensor

use_gpu = torch.cuda.is_available()


def outlier_detector(av, method='svm2', parameter=0.05):  # only for get_av in opensmax
    clf = []
    scaler = []
    for i in range(len(av)):
        if len(av[i]) <= 10:
            clf.append(None)
            scaler.append(None)
        else:
            av_t = np.array(av[i])
            if method == 'svm2' or method == 'svmn' or method == 'svm2n' or method == 'svm1':
                clf_t = svm.OneClassSVM(nu=parameter, kernel="linear")
            else:
                clf_t = IsolationForest(n_estimators=10)
            av_t = av_t.reshape((-1, av_t.shape[-1]))
            scaler_t = MinMaxScaler()
            scaler_t.fit(av_t)
            if method == 'svmn' or method == 'svm2n':
                clf_t.fit(av_t)
            else:
                clf_t.fit(scaler_t.transform(av_t))
            clf.append(clf_t)
            scaler.append(scaler_t)
    return clf, scaler


def get_av(MODEL_DIR, method):  # 返回每个类别 训练集的均值、方差和训练向量, only for opensmax and doc
    openmax_train = np.load(MODEL_DIR + 'cls_activations_train.npy')
    y_predict = np.load(MODEL_DIR + 'cls_predictions_train.npy')
    softmax_train = np.load(MODEL_DIR + 'cls_soft_activations_train.npy')
    y_true = np.load(MODEL_DIR + 'label_training.npy')
    class_num = openmax_train.shape[-1]

    res = []
    for i in range(class_num):
        res.append([])

    for i in range(openmax_train.shape[0]):
        if y_predict[i] == y_true[i]:
            openmax_av = openmax_train[i]
            softmax_av = softmax_train[i]
            label = int(y_true[i])
            if method == 'opensmax':
                res[label].append(np.array([openmax_av[label], softmax_av[label]]))
            elif 'doc' in method:
                res[label].append(openmax_av[label])
    for i in range(class_num):
        res[i] = np.array(res[i])
    mav = []
    vav = []
    for i in range(class_num):
        mav.append([np.mean(res[i], axis=0)])
        vav.append([np.var(res[i], axis=0)])

    if method == 'opensmax':
        clf, scaler = outlier_detector(res, method='svm2')
        return clf, scaler
    else:
        return mav, res, vav


def get_av_new(openmax_train, y_predict, softmax_train, y_true,
               method):  # 返回每个类别 训练集的均值、方差和训练向量, only for opensmax and doc
    class_num = openmax_train.shape[-1]

    res = []
    for i in range(class_num):
        res.append([])

    for i in range(openmax_train.shape[0]):
        if y_predict[i] == y_true[i]:
            openmax_av = openmax_train[i]
            softmax_av = softmax_train[i]
            label = int(y_true[i])
            if method == 'opensmax':
                res[label].append(np.array([openmax_av[label], softmax_av[label]]))
            elif 'doc' in method:
                res[label].append(openmax_av[label])
    for i in range(class_num):
        res[i] = np.array(res[i])
    mav = []
    vav = []
    for i in range(class_num):
        mav.append([np.mean(res[i], axis=0)])
        vav.append([np.var(res[i], axis=0)])

    if method == 'opensmax':
        clf, scaler = outlier_detector(res, method='svm2')
        return clf, scaler
    else:
        return mav, res, vav


def get_mean(MODEL_DIR):  # for center_ii
    cls2_activations_known_pos_train = np.load(MODEL_DIR + 'cls_activations_known_pos_train.npy')
    cls2_activations_known_neg_train = np.load(MODEL_DIR + 'cls_activations_known_neg_train.npy')
    mean_0 = np.mean(cls2_activations_known_pos_train, axis=0)
    mean_1 = np.mean(cls2_activations_known_neg_train, axis=0)
    return mean_0, mean_1


def get_mean_new(cls_activations_train, known_label_train):  # for center_ii
    mean = []
    for i in range(len(np.unique(known_label_train))):
        mean.append(np.mean(cls_activations_train[known_label_train == i], axis=0))
    return np.array(mean)


def create_openmaxevm(MODEL_DIR):
    activations_train = np.load(MODEL_DIR + 'cls_activations_train.npy')
    predictions_train = np.load(MODEL_DIR + 'cls_predictions_train.npy')
    label_training = np.load(MODEL_DIR + 'label_training.npy')
    mean_activations, channel_distances = evt_fitting.compute_mav_distances(activations_train, predictions_train,
                                                                            label_training)
    # print(len(mean_activations),len(channel_distances))
    label = list(range(len(activations_train[0])))
    weibull_model = evt_fitting.pgl_weibull_tailfitting(channel_distances, mean_activations, labellist=label,
                                                        tailsize=20)
    return weibull_model


def openmax_evm(activations_train, predictions_train, label_training):
    mean_activations, channel_distances = evt_fitting.compute_mav_distances(activations_train, predictions_train,
                                                                            label_training)
    # print(len(mean_activations),len(channel_distances))
    label = list(range(len(np.unique(label_training))))
    weibull_model = evt_fitting.pgl_weibull_tailfitting(channel_distances, mean_activations, labellist=label,
                                                        tailsize=20)
    return weibull_model


def create_openidsevm(MODEL_DIR):  # for openids series
    cls2_activations_known_pos_train = np.load(MODEL_DIR + 'cls_activations_known_pos_train.npy')
    cls2_activations_known_neg_train = np.load(MODEL_DIR + 'cls_activations_known_neg_train.npy')
    mr1 = libmr.MR()
    mr2 = libmr.MR()
    mr1.fit_low(cls2_activations_known_pos_train[:, 0], len(cls2_activations_known_pos_train))
    mr2.fit_low(cls2_activations_known_neg_train[:, 1], len(cls2_activations_known_neg_train))
    return mr1, mr2


def create_save_openidsevm(MODEL_DIR, cls2_activations_known_pos_train, known_label_train):  # for openids series
    for i in range(len(np.unique(known_label_train))):
        mr = libmr.MR()
        fit_data = cls2_activations_known_pos_train[known_label_train == i]
        mr.fit_low(fit_data[:, i], len(fit_data))
        saved = str(mr)
        fh = open(MODEL_DIR + str(i) + 'weibull_model.txt', 'w')
        fh.write(saved)
        fh.close()


def load_openidsevm(MODEL_DIR, class_num):  # for openids series
    mr_array = []
    for i in range(class_num):
        with open(MODEL_DIR + str(i) + 'weibull_model.txt') as f:
            read_data = f.read()
        f.closed
        mr_array.append(libmr.load_from_string(read_data))
    return mr_array


def anomaly_class(time_dir, known_data_train, known_label_train):
    clf = OneClassSVM(gamma='auto').fit(known_data_train)
    joblib.dump(clf, time_dir + '/ocsvm.pkl')  # 读取Model clf3 = joblib.load('save/clf.pkl')


def anomaly_perclass(times1_dir, data_train):
    known_data_train = data_train[:, :-1]
    known_label_train = data_train[:, -1]
    all_class = np.unique(known_label_train)
    for cls in range(len(all_class)):
        data_class = known_data_train[known_label_train == all_class[cls]]
        data_class = shuffle_tensor(data_class)

        if len(data_class) > 80000:
            ell = EllipticEnvelope(contamination=0.01, random_state=0).fit(data_class[:80000])
            iso = IsolationForest(random_state=0, n_estimators=250, contamination=0.01).fit(data_class[:80000])
        else:
            ell = EllipticEnvelope(contamination=0.01, random_state=0).fit(data_class)
            iso = IsolationForest(random_state=0, n_estimators=250, contamination=0.01).fit(data_class)
        joblib.dump(ell, times1_dir + 'ell' + str(cls) + '.pkl')  # 读取Model ell = joblib.load('save/ell.pkl')
        joblib.dump(iso, times1_dir + 'iso' + str(cls) + '.pkl')  # 读取Model ell = joblib.load('save/ell.pkl')

        if len(data_class) > 10000:
            ocsvm = OneClassSVM(gamma='auto').fit(data_class[:10000])
            lof = LocalOutlierFactor(contamination=0.01, novelty=True).fit(data_class[:10000])
        else:
            ocsvm = OneClassSVM(gamma='auto').fit(data_class)
            lof = LocalOutlierFactor(contamination=0.01, novelty=True).fit(data_class)
        joblib.dump(ocsvm, times1_dir + 'ocsvm' + str(cls) + '.pkl')  # 读取Model ell = joblib.load('save/ell.pkl')
        joblib.dump(lof, times1_dir + 'lof' + str(cls) + '.pkl')  # 读取Model ell = joblib.load('save/ell.pkl')


def anomaly_detect_perclass(times1_dir, data_train, data_test):
    known_label_train = data_train[:, -1]
    all_class = np.unique(known_label_train)
    results = []
    for cls in range(len(all_class)):
        ell = joblib.load(times1_dir + 'ell' + str(cls) + '.pkl')
        result_ell_cls = ell.predict(data_test)

        ocsvm = joblib.load(times1_dir + 'ocsvm' + str(cls) + '.pkl')
        result_ocsvm_cls = ocsvm.predict(data_test)

        iso = joblib.load(times1_dir + 'iso' + str(cls) + '.pkl')
        result_iso_cls = iso.predict(data_test)

        lof = joblib.load(times1_dir + 'lof' + str(cls) + '.pkl')
        result_lof_cls = lof.predict(data_test)
        results.append([result_ell_cls, result_ocsvm_cls, result_iso_cls, result_lof_cls])
    np.save(times1_dir + 'anomaly_results.npy', np.array(results))


def anomaly_softmax_pred(time_dir, threshold, cls2feature_test):
    clf = joblib.load(time_dir + '/ocsvm.pkl')
    ano_pred = clf.predict(cls2feature_test)
    max_y = np.max(np.array(cls2feature_test), axis=1)
    pred_y = torch.argmax(torch.Tensor(np.array(cls2feature_test)), dim=1).numpy()
    for i in range(len(cls2feature_test)):
        if max_y[i] < threshold or ano_pred[i] == -1:
            pred_y[i] = len(cls2feature_test[0])
    return pred_y


def peranomaly_softmax_pred(threshold, cls2feature_test, results, mode='softmax'):
    max_y = np.max(np.array(cls2feature_test), axis=1)
    pred_y = torch.argmax(torch.Tensor(np.array(cls2feature_test)), dim=1).numpy()
    for i in range(len(cls2feature_test)):
        if max_y[i] < threshold or results[i] == -1:
            if mode == 'softmax':
                pred_y[i] = len(cls2feature_test[0])
            else:
                pred_y[i] = len(cls2feature_test[0]) - 1
    return pred_y


def openmax_pred(weibull_model, thresholds, activations):
    pred = []
    for i in range(len(activations)):
        temp = evt_fitting.recalibrate_scores(weibull_model, activations[i], alpharank=len(activations[i]))
        pred.append(temp)
    print('len(pred[0])',len(pred[0]))
    pred_y = [softmax_pred(threshold, pred, mode='openmax') for threshold in thresholds]
    return pred_y


def crosr_pred(weibull_model, thresholds, activations, hiddens):
    pred = []
    for i in range(len(activations)):
        temp = evt_fitting.crosr_recalibrate_scores(weibull_model, activations[i], hiddens[i],
                                                    alpharank=len(activations[i]))
        pred.append(temp)
    pred_y = [softmax_pred(threshold, pred, mode='openmax') for threshold in thresholds]
    return pred_y


def peranomaly_openmax_pred(weibull_model, thresholds, activations, anomaly_results):
    pred = []
    for i in range(len(activations)):
        temp = evt_fitting.recalibrate_scores(weibull_model, activations[i], alpharank=len(activations[i]))
        pred.append(temp)
    print('peranomaly len(pred[0])',len(pred[0]))
    pred_y = [peranomaly_softmax_pred(threshold, pred, anomaly_results, mode='openmax') for threshold in thresholds]
    return pred_y


def crosr_openmax_pred(weibull_model, thresholds, activations, hiddens, anomaly_results):
    pred = []
    for i in range(len(activations)):
        temp = evt_fitting.crosr_recalibrate_scores(weibull_model, activations[i], hiddens[i],
                                                    alpharank=len(activations[i]))
        pred.append(temp)
    pred_y = [peranomaly_softmax_pred(threshold, pred, anomaly_results, mode='openmax') for threshold in thresholds]
    return pred_y


def opensmax_pred(openmax_test, softmax_test, clf, scaler):
    pred_y2 = torch.argmax(torch.Tensor(np.array(softmax_test)), dim=1).numpy()

    for i in range(len(softmax_test)):
        label = pred_y2[i]
        tmp = np.array([openmax_test[i][label], softmax_test[i][label]]).reshape(1, 2)
        opensmax_clf = scaler[label]
        if opensmax_clf is None:
            is_unknown = False
        else:
            tmp = opensmax_clf.transform(tmp)
            is_unknown = (clf[label].predict(tmp)[0] == -1)
        if is_unknown:
            pred_y2[i] = len(softmax_test[0])
    return pred_y2


def softmax_pred(threshold, cls2feature_test, mode='softmax'):
    max_y = np.max(np.array(cls2feature_test), axis=1)
    pred_y = torch.argmax(torch.Tensor(np.array(cls2feature_test)), dim=1).numpy()
    for i in range(len(cls2feature_test)):
        if max_y[i] < threshold:
            if mode == 'softmax':
                pred_y[i] = len(cls2feature_test[0])
            else:
                pred_y[i] = len(cls2feature_test[0]) - 1
    return pred_y


def peranomaly_opensmax_pred(openmax_test, softmax_test, clf, scaler, anomaly_results):
    pred_y2 = torch.argmax(torch.Tensor(np.array(softmax_test)), dim=1).numpy()
    for i in range(len(softmax_test)):
        label = pred_y2[i]
        tmp = np.array([openmax_test[i][label], softmax_test[i][label]]).reshape(1, 2)
        opensmax_clf = scaler[label]
        if opensmax_clf is None:
            is_unknown = False
        else:
            tmp = opensmax_clf.transform(tmp)
            is_unknown = (clf[label].predict(tmp)[0] == -1)
        if is_unknown or anomaly_results[i] == -1:
            pred_y2[i] = len(softmax_test[0])
    return pred_y2


def peranomaly_doc_pred(features, mav, vav, anomaly_results):
    mav = np.array(mav)
    vav = np.array(vav)
    pred_y2 = torch.argmax(torch.Tensor(np.array(features)), dim=1).numpy()
    for i in range(len(features)):
        label = pred_y2[i]
        if (features[i][label] < 0.5 or features[i][label] < mav[label] - 3 * vav[label]) or anomaly_results[i] == -1:
            pred_y2[i] = len(features[0])
    return pred_y2


def doc_pred(features, mav, vav):
    mav = np.array(mav)
    vav = np.array(vav)
    pred_y2 = torch.argmax(torch.Tensor(np.array(features)), dim=1).numpy()
    for i in range(len(features)):
        label = pred_y2[i]
        if features[i][label] < 0.5 or features[i][label] < mav[label] - 3 * vav[label]:
            pred_y2[i] = len(features[0])
    return pred_y2


def peranomaly_centerii_pred(threshold, cls2feature_test, mean, anomaly_results):
    pred_y2 = torch.argmax(torch.Tensor(np.array(cls2feature_test)), dim=1).numpy()
    for i in range(len(cls2feature_test)):
        dis = np.array([np.linalg.norm(cls2feature_test[i] - mean[j]) for j in range(len(mean))])
        if min(dis) > threshold or anomaly_results[i] == -1:
            pred_y2[i] = len(cls2feature_test[0])
    return pred_y2


def centerii_pred(threshold, cls2feature_test, mean):
    pred_y2 = torch.argmax(torch.Tensor(np.array(cls2feature_test)), dim=1).numpy()
    for i in range(len(cls2feature_test)):
        dis = np.array([np.linalg.norm(cls2feature_test[i] - mean[j]) for j in range(len(cls2feature_test[0]))])
        if min(dis) > threshold:
            pred_y2[i] = len(cls2feature_test[0])
    return pred_y2


def peranomaly_openids_pred(evms, threshold, cls2feature_test, anomaly_results):
    not_y = np.array(
        [evms[i].w_score_vector(np.double(np.array(cls2feature_test[:, 0]))) for i in range(len(cls2feature_test))])
    pred_y2 = torch.argmax(torch.Tensor(np.array(cls2feature_test)), dim=1).numpy()
    for i in range(len(cls2feature_test)):
        if min(not_y[i]) > threshold or anomaly_results[i] == -1:
            pred_y2[i] = len(cls2feature_test[0])
        return pred_y2


def openids_pred(evms, threshold, cls2feature_test):
    not_y = np.array(
        [evms[i].w_score_vector(np.double(np.array(cls2feature_test[:, 0]))) for i in range(len(cls2feature_test))])
    pred_y2 = torch.argmax(torch.Tensor(np.array(cls2feature_test)), dim=1).numpy()
    for i in range(len(cls2feature_test)):
        if min(not_y[i]) > threshold:
            pred_y2[i] = len(cls2feature_test[0])
    return pred_y2


def cvae_pred(evm, threshold, pred, x_recon_loss):
    y = evm.w_score_vector(np.double(np.array(x_recon_loss)))
    pred_y = torch.argmax(torch.Tensor(np.array(pred)), dim=1).numpy()
    for i in range(len(pred)):
        if y[i] < threshold:
            pred_y[i] = len(pred[0])
    return pred_y


def peranomaly_cvae_pred(evm, threshold, pred, x_recon_loss, anomaly_results):
    y = evm.w_score_vector(np.double(np.array(x_recon_loss)))
    pred_y = torch.argmax(torch.Tensor(np.array(pred)), dim=1).numpy()
    for i in range(len(pred_y)):
        if y[i] < threshold or anomaly_results[i] == -1:
            pred_y[i] = len(pred[0])
    return pred_y


def gmm_pred(threshold, probs, class_predict):
    import copy
    b = copy.deepcopy(class_predict)
    max_y = np.max(np.array(probs), axis=1)
    for i in range(len(probs)):
        if max_y[i] < threshold:
            b[i] = len(probs[0])
    return b


def peranomaly_gmm_pred(threshold, probs, class_predict, anomaly_results):
    max_y = np.max(np.array(probs), axis=1)
    for i in range(len(probs)):
        if max_y[i] < threshold or anomaly_results[i] == -1:
            class_predict[i] = len(probs[0])
    return class_predict


def save_clf_scaler(clf, scaler, class_num, MODEL_DIR):
    for category in range(class_num):
        joblib.dump(clf[category], MODEL_DIR + 'clf' + str(category) + '.pkl')
        joblib.dump(scaler[category], MODEL_DIR + 'scaler' + str(category) + '.pkl')


def load_clf_scaler(class_num, MODEL_DIR):
    clf = []
    scaler = []
    for category in range(class_num):
        clf.append(joblib.load(MODEL_DIR + 'clf' + str(category) + '.pkl'))
        scaler.append(joblib.load(MODEL_DIR + 'scaler' + str(category) + '.pkl'))
    return clf, scaler
