import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import pickle
from main_0124 import get_thresholds

font1 = {'family': 'Times New Roman',
         'size': 20,
         }


def plot_conf_mat(cm, save_dir, labels):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(len(labels) * 2, len(labels) * 2))
    disp.plot(ax=ax, values_format='.0f')
    plt.savefig(save_dir)
    plt.show()


def plot_conf_mat1(cm, save_dir, labels):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(len(labels) * 2, len(labels) * 2))
    disp.plot(ax=ax)
    plt.savefig(save_dir)
    plt.show()


def plot_figures(data, save_dir, methods, colous, file):
    plt.figure(figsize=(10, 8))  # 设置图大小 figsize=(4,3)
    plt.title(file[:-8])
    plt.xlabel('thresholds')
    plt.ylabel('detection ratio')
    for i, method in enumerate(methods):
        if method == 'center_ii':
            plt.plot(data[i][0][-101:], np.array(data[i][2][-101:]) - np.array(data[i][1][-101:]), c=colous[i],
                     marker="^", label=methods[i], markevery=5, alpha=0.7)
        else:
            plt.plot(data[i][0], np.array(data[i][2]) - np.array(data[i][1]), c=colous[i], marker="^", label=methods[i],
                     markevery=5, alpha=0.5)
    plt.legend(loc='best')
    plt.savefig(save_dir)
    plt.show()


def plot_figures1(data, save_dir, methods, file):
    methods = ['Baseline', 'OpenMax', 'DOC', 'Center_IDS', 'Center_II', 'CVAE_EVT', 'CROSR']
    plt.figure(figsize=(10, 8))  # 设置图大小 figsize=(4,3)
    #plt.title(file[:-8])
    plt.xlabel('methods',font1)
    plt.ylabel('detected num',font1)
    bar_width = 0.40
    plt.bar(np.arange(len(methods)), data[0], label='oc', color='steelblue', alpha=0.8, width=bar_width)
    plt.bar(np.arange(len(methods)) + bar_width, data[1], label='oc*', color='indianred', alpha=0.8, width=bar_width)
    tex_height = max(np.array(data[1])) * 0.01
    plt.tick_params(labelsize=15)
    for x, y in enumerate(data[0]):
        plt.text(x,  y + tex_height, '%s' % y,font1, ha='center')

    for x, y in enumerate(data[1]):
        plt.text( x + bar_width, y + tex_height, '%s' % y,font1, ha='center')

    plt.xticks(np.arange(len(methods)) + 0.20, methods,size=15,family='Times New Roman')
    plt.legend(loc='best',prop=font1)
    plt.savefig(save_dir)
    plt.show()


def plot_conmat(path, methods):
    re_best_results = np.load(path + methods + '/re_best_results.npy', allow_pickle=True)
    known_classes = list(np.load(path + 'known_classes.npy', allow_pickle=True))
    known_classes.append('Unknwon')
    plot_conf_mat(re_best_results[0][2][15], path + '/images/' + methods + '_conf_close.pdf', known_classes)
    plot_conf_mat(re_best_results[1][2][15], path + '/images/' + methods + '_conf_open.pdf', known_classes)
    plot_conf_mat(re_best_results[2][2][15], path + '/images/' + methods + '_conf_openplus.pdf', known_classes)


def plot_adv(path, file, methods, colous):
    data = []
    for method in methods:
        this_data = [get_thresholds(method)]
        open_file = open(path + method + '/' + file, 'rb')
        open_data = pickle.load(open_file)
        x1 = [d[1] for d in open_data]
        x2 = [d[2] for d in open_data]
        this_data.append(x1)
        this_data.append(x2)
        open_file.close()
        data.append(this_data)
    plot_figures(data, path + '/images/' + file + '.pdf', methods, colous, file)


def plot_adv1(path, file, methods):
    data = []
    data1 = []
    data2 = []
    for method in methods:
        re_best_results = np.load(path + method + '/re_best_results.npy', allow_pickle=True)
        index = get_thresholds(method).index(re_best_results[1][1])
        open_file = open(path + method + '/' + file, 'rb')
        open_data = pickle.load(open_file)
        x1 = open_data[index][1]
        x2 = open_data[index][2]
        data1.append(x1)
        data2.append(x2)
        open_file.close()
    data.append(data1)
    data.append(data2)
    plot_figures1(data, path + '/images/' + file + '.pdf', methods, file)


methods = ['softmax', 'openmax', 'doc', 'center_ids', 'center_ii', 'cvae_evt', 'crosr']
colours = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan']
import os

for num in range(1):
    for time in range(1):
        time = 0
        num = 1
        save_path = 'D:/ids2017/model/times_' + str(time) + '/' + 'unum' + str(num + 1) + '/'
        imgeges_DIR = save_path + 'images' + '/'
        isExists = os.path.exists(imgeges_DIR)
        if not isExists:
            os.makedirs(imgeges_DIR)
        '''
        for method in methods:
            try:
                plot_conmat(save_path, method)
            except:
                print('false 0')


        hard_results = np.load(save_path + '/hard_detect.npy', allow_pickle=True)
        hard = [
            [str(s[0][0]) + '/' + str(s[0][1]), str(s[1][0]) + '/' + str(s[1][1]), str(s[2][0]) + '/' + str(s[2][1]),
             str(s[3][0]) + '/' + str(s[3][1]), str(s[4][0]) + '/' + str(s[4][1]), str(s[5][0]) + '/' + str(s[5][1]),
             str(s[6][0]) + '/' + str(s[6][1])] for s in hard_results]

        plot_conf_mat1(np.array(hard),imgeges_DIR+'hard.pdf', methods)


        '''
        for file in ['real_adv_dec.txt.pkl', 'noise_gussian_adv_dec.txt.pkl', 'noise_uniform_adv_dec.txt.pkl']:
            plot_adv1(save_path, file, methods)
            '''
            try:
                
            except:
                print('false')
            '''