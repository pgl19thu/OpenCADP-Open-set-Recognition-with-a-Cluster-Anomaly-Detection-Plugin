import joblib
import libmr
import numpy as np
import torch
from sklearn.mixture import GaussianMixture as GMM
import model
from evt_fitting import save_weibull_model, load_weibull_model
from main_anoamly import anomaly_predict
from predict_function import get_av_new, openmax_evm, save_clf_scaler, \
    get_mean_new, load_clf_scaler, peranomaly_openmax_pred, peranomaly_gmm_pred, peranomaly_openids_pred, \
    peranomaly_cvae_pred, crosr_openmax_pred, crosr_pred
from predict_function import softmax_pred, openmax_pred, opensmax_pred, doc_pred, centerii_pred, openids_pred, \
    cvae_pred, gmm_pred, peranomaly_softmax_pred, create_save_openidsevm, load_openidsevm, peranomaly_doc_pred, \
    peranomaly_centerii_pred, peranomaly_opensmax_pred
use_gpu = torch.cuda.is_available()


def create_cls(data_training, label_training, cls_path, size_input=71, size_output=2, LR_cls=0.005, Batch_cls=64,
               EPOCH_cls=5, method='openids'):
    if method == 'doc':
        net = model.Doc(size_input=size_input, size_output=size_output)
    elif method == 'crosr':
        net = model.crosr_net(size_input=size_input, size_output=size_output)
    else:
        net = model.Normal(size_input=size_input, size_output=size_output)

    if use_gpu:  # 1
        net = net.cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=LR_cls)
    if method == 'doc':
        model.training_doc(net, optimizer, data_training, label_training, EPOCH_cls, Batch_cls, cls_path)
    elif method == 'crosr':
        model.training_crosr(net, optimizer, data_training, label_training, EPOCH_cls, Batch_cls, cls_path)
    else:
        model.training_class(net, optimizer, data_training, label_training, EPOCH_cls, Batch_cls, cls_path, method)
    return net


def create_cvae_cls(data_training, cls_path, size_input=71, size_output=2, size_hidden=3, LR_cls=0.005, Batch_cls=64,
                    EPOCH_cls=1):
    net = model.CvaeEvt(size_input=size_input, size_output=size_output, size_hidden=size_hidden)
    if use_gpu:  # 1
        net = net.cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=LR_cls, momentum=0.9)
    model.training_cvae_cls(net, optimizer, data_training, EPOCH_cls, Batch_cls, cls_path)
    return net


def train_openids(MODEL_DIR, data_train, method):
    known_label_train = data_train[:, -1]
    known_data_train = data_train[:, :-1]
    '''--training cls and save features based on cls to MODEL_DIR--'''
    """--create  model path--"""
    cls_path = MODEL_DIR + 'cls.pkl'
    print('--training cls and save features based on cls to MODEL_DIR--')
    model_cls = create_cls(known_data_train, known_label_train, cls_path, size_input=len(known_data_train[0]),
                           size_output=len(np.unique(known_label_train)), EPOCH_cls=5, method=method)
    if method == 'softmax':
        import adv_test
        adv_scale = 0.05
        real_target = adv_test.get_advSamples(model_cls, adv_scale, data_train)
        np.save(MODEL_DIR + 'real_target.npy', real_target)
        noise_target = adv_test.get_advSamples_random(model_cls, adv_scale, len(data_train))
        np.save(MODEL_DIR + 'noise_target.npy', noise_target)
    if method == 'crosr':
        y_, z_, x_ = model_cls(torch.Tensor(known_data_train).cuda())
        predicts = np.argmax(y_.data.cpu().numpy(), 1)
        weibull_model = openmax_evm(z_.data.cpu().numpy(), predicts, known_label_train)
        save_weibull_model(weibull_model, len(np.unique(known_label_train)), MODEL_DIR)
    else:
        cls_activations_train, cls_predictions_train, cls_soft_activations_train = model.get_cls_act_pre_soft(
            known_data_train, cls_path)

    if method in ['openmax', 'center_ids']:
        weibull_model = openmax_evm(cls_activations_train, cls_predictions_train, known_label_train)
        save_weibull_model(weibull_model, len(np.unique(known_label_train)), MODEL_DIR)
    if method in ['opensmax']:
        clf, scaler = get_av_new(cls_activations_train, cls_predictions_train, cls_soft_activations_train,
                                 known_label_train, method)
        save_clf_scaler(clf, scaler, len(np.unique(known_label_train)), MODEL_DIR)
    if method in ['doc']:
        mav, res, vav = get_av_new(cls_activations_train, cls_predictions_train, cls_soft_activations_train,
                                   known_label_train, method)
        np.save(MODEL_DIR + 'mav.npy', np.array(mav))
        np.save(MODEL_DIR + 'res.npy', np.array(res))
        np.save(MODEL_DIR + 'vav.npy', np.array(vav))
    if method in ['center_ii', 'center_ocn']:
        mean = get_mean_new(cls_activations_train, known_label_train)
        np.save(MODEL_DIR + 'mean.npy', np.array(mean))
    if method == 'GMM':
        gmm = GMM(n_components=len(np.unique(known_label_train))).fit(cls_activations_train)
        joblib.dump(gmm, MODEL_DIR + 'gmm' + '.pkl')
    if method in ['openids', 'openids_noae']:
        create_save_openidsevm(MODEL_DIR, cls_activations_train, known_label_train)

    return model_cls


def test_openids(known_classes, time_dir, MODEL_DIR, test_data, method, thresholds, class_num):
    """get test features"""
    """get  model path"""
    cls_path = MODEL_DIR + 'cls.pkl'
    if method == 'crosr':
        model_cls = torch.load(cls_path)
        y_, z_, x_ = model_cls(torch.Tensor(test_data).cuda())
        class_predict = np.argmax(y_.data.cpu().numpy(), 1)
    else:
        cls_features, class_predict, softmax_predict = model.get_cls_act_pre_soft(test_data, cls_path)

    anomaly_results = anomaly_predict(time_dir, test_data, known_classes[class_predict])
    print('end of getting anomaly_results')

    if method in ['openmax', 'center_ids']:
        evms = load_weibull_model(class_num, MODEL_DIR)
        predict_test = openmax_pred(evms, thresholds, cls_features)
        predict_test1 = peranomaly_openmax_pred(evms, thresholds, cls_features, anomaly_results[-1])
        predict_tests = [[thresholds[i], class_predict, predict_test[i], predict_test1[i]] for i in
                         range(len(thresholds))]
    elif method in ['crosr']:
        evms = load_weibull_model(class_num, MODEL_DIR)
        predict_test = crosr_pred(evms, thresholds, y_.data.cpu().numpy(), z_.data.cpu().numpy())
        predict_test1 = crosr_openmax_pred(evms, thresholds, y_.data.cpu().numpy(), z_.data.cpu().numpy(),
                                           anomaly_results[-1])
        predict_tests = [[thresholds[i], class_predict, predict_test[i], predict_test1[i]] for i in
                         range(len(thresholds))]
    else:
        predict_tests = []
        for threshold in thresholds:
            predict_anos = [threshold, class_predict]
            if method == 'doc':
                mav = np.load(MODEL_DIR + 'mav.npy')
                vav = np.load(MODEL_DIR + 'vav.npy')
                predict_test = doc_pred(cls_features, mav, vav)
                predict_anos.append(predict_test)
                predict_anos.append(peranomaly_doc_pred(cls_features, mav, vav, anomaly_results[-1]))
            elif method in ['center_ii', 'center_ocn']:
                mean = np.load(MODEL_DIR + 'mean.npy')
                predict_test = centerii_pred(threshold, cls_features, mean)
                predict_anos.append(predict_test)
                predict_anos.append(peranomaly_centerii_pred(threshold, cls_features, mean, anomaly_results[-1]))
            elif method == 'opensmax':
                clf, scaler = load_clf_scaler(class_num, MODEL_DIR)
                print(len(clf), len(scaler))
                predict_test = opensmax_pred(cls_features, softmax_predict, clf, scaler)
                predict_anos.append(predict_test)
                predict_anos.append(
                    peranomaly_opensmax_pred(cls_features, softmax_predict, clf, scaler, anomaly_results[-1]))
            elif method in ['openids_nopevm', 'softmax']:
                predict_test = softmax_pred(threshold, softmax_predict)
                predict_anos.append(predict_test)
                predict_anos.append(peranomaly_softmax_pred(threshold, softmax_predict, anomaly_results[-1]))
            elif method == 'GMM':
                gmm = joblib.load(MODEL_DIR + 'gmm' + '.pkl')
                probs = gmm.predict_proba(cls_features)
                predict_test = gmm_pred(threshold, probs, class_predict)
                predict_anos.append(predict_test)
                predict_anos.append(peranomaly_gmm_pred(threshold, probs, class_predict, anomaly_results[-1]))
            else:
                evms = load_openidsevm(MODEL_DIR, class_num)
                predict_test = openids_pred(evms, threshold, cls_features)
                predict_anos.append(predict_test)
                predict_anos.append(peranomaly_openids_pred(evms, threshold, cls_features, anomaly_results[-1]))
            predict_tests.append(predict_anos)
    return predict_tests


def train_cvae_evt(MODEL_DIR, known_data_train):
    """create  model path"""
    cls_path = MODEL_DIR + 'cls.pkl'
    '''--training cls and save features based on cls to MODEL_DIR--'''
    print('--training cls and save features based on cls to MODEL_DIR--')
    model_cls = create_cvae_cls(known_data_train, cls_path, size_input=len(known_data_train[0]) - 1,
                                size_output=len(np.unique(known_data_train[:, -1])))
    training_recon_loss = model.get_output_cvae_evt(cls_path, known_data_train[:, :-1], known_data_train[:, -1], stau=2)

    mr = libmr.MR()
    mr.fit_low(training_recon_loss, 200)
    saved = str(mr)
    fh = open(MODEL_DIR + 'libmr_model.txt', 'w')
    fh.write(saved)
    fh.close()
    return model_cls


def test_cvae_evt(known_classes, time_dir, MODEL_DIR, test_data, thresholds):
    cls_path = MODEL_DIR + 'cls.pkl'
    predicts, test_recon_loss = model.get_output_cvae_evt(cls_path, test_data)
    class_predict = np.argmax(predicts, 1)
    anomaly_results = anomaly_predict(time_dir, test_data, known_classes[class_predict])
    with open(MODEL_DIR + 'libmr_model.txt') as f:
        read_data = f.read()
    f.closed
    mr = libmr.load_from_string(read_data)

    predict_tests = []
    for threshold in thresholds:
        predict_anos = [threshold, class_predict]
        predict_test = cvae_pred(mr, threshold, predicts, test_recon_loss)
        predict_anos.append(predict_test)
        predict_anos.append(peranomaly_cvae_pred(mr, threshold, predicts, test_recon_loss, anomaly_results[-1]))
        predict_tests.append(predict_anos)
    return predict_tests
