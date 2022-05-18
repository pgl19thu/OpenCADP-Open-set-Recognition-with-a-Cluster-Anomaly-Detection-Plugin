import os
import numpy as np
import get_data
import pickle
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

from main_anoamly import anomaly_detect
from train_test_function import train_openids, train_cvae_evt, test_openids, test_cvae_evt


def generate_unknown_class(num_unknown):
    import random
    class_attack = [['FTP-BruteForce', 'SSH-Bruteforce'],
                    ['DoS attacks-Hulk', 'DoS attacks-GoldenEye', 'DoS attacks-SlowHTTPTest', 'DoS attacks-Slowloris'],
                    ['DDOS attack-HOIC', 'DDOS attack-LOIC-UDP', 'DDoS attacks-LOIC-HTTP'],
                    ['Infilteration'],
                    ['Bot']]

    unknowns = random.sample(class_attack, num_unknown)
    unknown_classes = ['Brute Force -Web', 'Brute Force -XSS', 'SQL Injection']
    known_classes = ['Bengin']
    for s in class_attack:
        if s in unknowns:
            unknown_classes = unknown_classes + s
        else:
            known_classes = known_classes + s
    print('known_classes:', known_classes)
    print('unknown_classes:', unknown_classes)
    return known_classes, unknown_classes


def get_thresholds(method):
    if method in ['softmax', 'cvae_evt', 'openmax', 'center_ids', 'crosr']:
        tds = [i * 0.01 for i in range(101)]
    elif method in ['center_ii', 'center_ocn']:
        tds = [i * 0.01 for i in range(101)] + [1 + i * 0.1 for i in range(100)]
        tds.sort(reverse=True)
    else:
        tds = [0]  # if method in ['doc', 'GMM', 'opensmax']:
    return tds


def measures_of_acc_pre_rec_sup_confusion_macro_f1(actual_label, predict_label):
    acc_score = accuracy_score(actual_label, predict_label)
    confusion_score = confusion_matrix(actual_label, predict_label)
    pre, rec, f1, sup = precision_recall_fscore_support(actual_label, predict_label)
    macro_f1 = np.mean(f1)
    micro_f1 = f1_score(actual_label, predict_label, average='micro')
    return ['acc', acc_score, 'pre', pre, 'rec', rec, 'f1', f1, 'sup', sup, 'macro_f1', macro_f1, 'micro_f1', micro_f1,
            'confusion_matrix', confusion_score]


def save_result_txt(save_dir, save_data):
    f = open(save_dir, "w", encoding='utf-8')
    f.write(str([save_data]))
    f.close()
    output = open(save_dir + '.pkl', 'wb')
    pickle.dump(save_data, output, -1)
    output.close()


def cal_measure(MODEL_METHOD_DIR, actual_test, predict_test):
    '''性能计算'''
    results = []  # [ [th,re1,re2,...re7]    ]
    for i, single_predict_test in enumerate(predict_test):
        result = [single_predict_test[0]]
        for j in range(len(single_predict_test) - 1):
            result.append(measures_of_acc_pre_rec_sup_confusion_macro_f1(actual_test, single_predict_test[j + 1]))
        results.append(result)

    re_results = []  # [re1,re2..re7]
    re_best_results = []  # [best_re1,best_re2..best_re7]
    best_th = []  # 每个方法的最佳阈值  6*1    [th1,th2,..th7]
    for i in range(len(predict_test[0]) - 1):  # 每个方法，目前3个
        result_i = [[result[i + 1][11], result[0], result[i + 1]] for result in results]
        array_i = [result[i + 1][11] for result in results]
        re_results.append(result_i)
        max_i = np.argmax(np.array(array_i))
        re_best_results.append(result_i[max_i])
        best_th.append(max_i)
    save_result_txt(MODEL_METHOD_DIR + 're_results.txt', re_results)
    save_result_txt(MODEL_METHOD_DIR + 're_best_results.txt', re_best_results)

    return re_results, re_best_results, np.array(best_th)


def cal_hard_measure(actual_test, best_results, known_class_num):
    detect = []
    for i in range(len(best_results)):
        i1 = actual_test == known_class_num
        i2 = best_results[i][0] != known_class_num
        index_i = np.array([i1[i] and i2[i] for i in range(len(i1))])  # 被误识别为已知的未知索引
        detect_i = []
        for j in range(len(best_results)):
            detect_i.append([np.sum(np.array(best_results[j][0][index_i]) == known_class_num),
                             np.sum(np.array(best_results[j][1][index_i]) == known_class_num)])
        detect.append(detect_i)
    return np.array(detect)  # 4*4*2   方法*方法*[方法,方法 +ano]


if __name__ == '__main__':
    print("设置训练/测试参数")
    is_train = False  # False#
    is_test = True
    is_adv = True
    is_noise = False
    is_hard = False

    predict_data = []
    methods_cls = ['cvae_evt', 'openmax', 'crosr', 'softmax', 'center_ii', 'center_ids', 'doc']
    dict_class = {'Benign': 0, 'Bot': 1, 'Brute Force -Web': 2, 'Brute Force -XSS': 3, 'DDOS attack-HOIC': 4,
                  'DDOS attack-LOIC-UDP': 5, 'DDoS attacks-LOIC-HTTP': 6, 'DoS attacks-GoldenEye': 7,
                  'DoS attacks-Hulk': 8, 'DoS attacks-SlowHTTPTest': 9, 'DoS attacks-Slowloris': 10,
                  'FTP-BruteForce': 11, 'Infilteration': 12, 'SQL Injection': 13, 'SSH-Bruteforce': 14}

    print("设置数据,方法,模型参数")
    SAVE_DATA_DIR = 'osd/ids2018_train_data.npy'
    SAVE_TEST_DATA_DIR = 'osd/ids2018_test_data.npy'
    SAVE_DIR = 'model/'
    i = 0
    anomaly_dir = 'model/anomaly' + str(i) + '/'

    print("实验训练、测试开始...,进行5次")
    for times1 in range(5):
        times1 = 0
        times1_dir = SAVE_DIR + 'times_' + str(times1) + '/'
        for num_unknowns in range(1, 5):

            print("设置项目工作路径")
            WORK_DIR = times1_dir + 'unum' + str(num_unknowns) + '/'
            isExists = os.path.exists(WORK_DIR)
            if not isExists:
                print("工作路径不存在，建立项目工作路径，未知类数据，高斯噪声，统一噪声")
                os.makedirs(WORK_DIR)
                known_classes, unknown_class = generate_unknown_class(num_unknowns)
                np.save(WORK_DIR + 'unknown_class.npy', np.array(unknown_class))
                save_result_txt(WORK_DIR + 'unknown_class.txt', unknown_class)
                np.save(WORK_DIR + 'known_classes.npy', np.array(known_classes))
                save_result_txt(WORK_DIR + 'known_classes.txt', known_classes)

                noise_gussian = np.random.normal(0.5, 0.1, (500, 71))
                out = np.clip(noise_gussian, 0, 1.0)
                np.save(WORK_DIR + 'noise_gussian.npy', np.array(noise_gussian))
                save_result_txt(WORK_DIR + 'noise_gussian.txt', noise_gussian)

                noise_uniform = np.random.random((500, 71))
                np.save(WORK_DIR + 'noise_uniform.npy', np.array(noise_uniform))
                save_result_txt(WORK_DIR + 'noise_uniform.txt', noise_uniform)
                print("加载训练数据，设置项目及最佳结果保存列表")
                data_train, data_test, known_classes = get_data.get_train_test(unknown_class, SAVE_DATA_DIR,
                                                                               SAVE_TEST_DATA_DIR)
                anomaly_detect(anomaly_dir, data_train, known_classes, 0.001 * (i + 1))
            else:
                print("工作路径存在，直接加载未知类，高斯噪声，统一噪声")
                unknown_class = np.load(WORK_DIR + 'unknown_class.npy', allow_pickle=True)
                noise_gussian = np.load(WORK_DIR + 'noise_gussian.npy', allow_pickle=True)
                noise_uniform = np.load(WORK_DIR + 'noise_uniform.npy', allow_pickle=True)

                print("加载训练数据，设置项目及最佳结果保存列表")
                data_train, data_test, known_classes = get_data.get_train_test(unknown_class, SAVE_DATA_DIR,
                                                                               SAVE_TEST_DATA_DIR)

            results_times_num_unknown = []
            pred_best_methods = []
            for method in ['softmax', 'openmax', 'doc', 'center_ids', 'center_ii', 'cvae_evt',
                           'crosr']:  # 'cvae_evt','openmax','crosr','softmax','center_ii','center_ids','doc'
                print("设置方法阈值，工作路径", method)
                thresholds = get_thresholds(method)
                MODEL_METHOD_DIR = WORK_DIR + method + '/'
                isExists = os.path.exists(MODEL_METHOD_DIR)
                if not isExists:
                    os.makedirs(MODEL_METHOD_DIR)

                print("-----模型训练-----", method)
                if is_train:
                    print('------start training------')
                    if method in ['cvae_evt']:
                        model_cls = train_cvae_evt(MODEL_METHOD_DIR, data_train)
                    else:
                        model_cls = train_openids(MODEL_METHOD_DIR, data_train, method)
                    anomaly_detect(anomaly_dir, data_train, known_classes, 0.001 * (i + 1))
                print("-----模型测试-----", method)
                if is_test:
                    if method in ['cvae_evt']:
                        predict_test = test_cvae_evt(known_classes, anomaly_dir, MODEL_METHOD_DIR,
                                                     data_test[:, :-1], thresholds)
                    else:
                        predict_test = test_openids(known_classes, anomaly_dir, MODEL_METHOD_DIR,
                                                    data_test[:, :-1], method, thresholds, len(known_classes))

                    actual_test = data_test[:, -1]
                    re_result, re_best_result, best_ths = cal_measure(MODEL_METHOD_DIR, actual_test, predict_test)
                    pred_best_methods.append([predict_test[best_ths[1]][2], predict_test[best_ths[1]][-1]])  # n*2
                    # 某方法的结果，某方法+anoamly的结果
                    save_result_txt(MODEL_METHOD_DIR + 're_results.txt', re_result)
                    for z in re_best_result:
                        z[2].append('mean pre')
                        z[2].append(np.mean(z[2][3]))
                        z[2].append('mean recall')
                        z[2].append(np.mean(z[2][5]))
                    save_result_txt(MODEL_METHOD_DIR + 're_best_results.txt', re_best_result)
                    np.save(MODEL_METHOD_DIR + 're_results.npy', np.array(re_result))
                    np.save(MODEL_METHOD_DIR + 're_best_results.npy', np.array(re_best_result))
                    np.save(MODEL_METHOD_DIR + 'predict_test.npy', np.array(predict_test))

                print('-----对抗样本测试-----')
                if is_adv:
                    real_target = np.load(WORK_DIR + 'softmax' + '/' + 'real_target.npy')
                    noise_target = np.load(WORK_DIR + 'softmax' + '/' + 'noise_target.npy')
                    if len(real_target) != 0:
                        if method in ['cvae_evt']:
                            real_predict_test = test_cvae_evt(known_classes, anomaly_dir, MODEL_METHOD_DIR,
                                                              real_target, thresholds)
                        else:
                            real_predict_test = test_openids(known_classes, anomaly_dir, MODEL_METHOD_DIR,
                                                             real_target, method, thresholds, len(known_classes))
                        real_adv_dec = [[np.sum(i[1] == len(known_classes)), np.sum(i[2] == len(known_classes)),
                                         np.sum(i[-1] == len(known_classes))] for i in real_predict_test]
                        save_result_txt(MODEL_METHOD_DIR + 'real_adv_dec.txt', real_adv_dec)
                    if len(noise_target) != 0:
                        if method in ['cvae_evt']:
                            noise_predict_test = test_cvae_evt(known_classes, anomaly_dir, MODEL_METHOD_DIR,
                                                               noise_target, thresholds)
                        else:
                            noise_predict_test = test_openids(known_classes, anomaly_dir, MODEL_METHOD_DIR,
                                                              noise_target, method, thresholds, len(known_classes))
                        noise_adv_dec = [[np.sum(i[1] == len(known_classes)), np.sum(i[2] == len(known_classes)),
                                          np.sum(i[-1] == len(known_classes))] for i in noise_predict_test]
                        save_result_txt(MODEL_METHOD_DIR + 'noise_adv_dec.txt', noise_adv_dec)
                        # np.save(MODEL_METHOD_DIR + 'noise_adv_dec.npy', noise_adv_dec)
                print('-----噪声样本测试-----')
                if is_noise:
                    if method in ['cvae_evt']:
                        noise_gussian_predict_test = test_cvae_evt(known_classes, anomaly_dir,
                                                                   MODEL_METHOD_DIR, noise_gussian, thresholds)
                        noise_uniform_predict_test = test_cvae_evt(known_classes, anomaly_dir,
                                                                   MODEL_METHOD_DIR, noise_uniform, thresholds)
                    else:
                        noise_gussian_predict_test = test_openids(known_classes, anomaly_dir,
                                                                  MODEL_METHOD_DIR,
                                                                  noise_gussian, method, thresholds, len(known_classes))
                        noise_uniform_predict_test = test_openids(known_classes, anomaly_dir,
                                                                  MODEL_METHOD_DIR,
                                                                  noise_uniform, method, thresholds, len(known_classes))
                    print('len(noise_gussian_predict_test[0])', len(noise_gussian_predict_test[0]))
                    noise_gussian_adv_dec = [[np.sum(i[1] == len(known_classes)), np.sum(i[2] == len(known_classes)),
                                              np.sum(i[-1] == len(known_classes))] for i in noise_gussian_predict_test]
                    noise_uniform_adv_dec = [[np.sum(i[1] == len(known_classes)), np.sum(i[2] == len(known_classes)),
                                              np.sum(i[-1] == len(known_classes))] for i in noise_uniform_predict_test]
                    save_result_txt(MODEL_METHOD_DIR + 'noise_gussian_adv_dec.txt', noise_gussian_adv_dec)
                    save_result_txt(MODEL_METHOD_DIR + 'noise_uniform_adv_dec.txt', noise_uniform_adv_dec)

            print('-----困难样本测试-----')
            if is_hard:
                save_result_txt(WORK_DIR + 'pred_best_methods.txt', pred_best_methods)
                actual_test = data_test[:, -1]
                hard_detect = cal_hard_measure(actual_test, np.array(pred_best_methods),
                                               len(known_classes))
                save_result_txt(WORK_DIR + 'hard_detect.txt', hard_detect)
                np.save(WORK_DIR + 'hard_detect.npy', np.array(hard_detect))
