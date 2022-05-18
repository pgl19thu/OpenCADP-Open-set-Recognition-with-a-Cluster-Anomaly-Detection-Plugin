# -*- coding: utf-8 -*-

###################################################################################################
# Copyright (c) 2016 , Regents of the University of Colorado on behalf of the University          #
# of Colorado Colorado Springs.  All rights reserved.                                             #
#                                                                                                 #
# Redistribution and use in source and binary forms, with or without modification,                #
# are permitted provided that the following conditions are met:                                   #
#                                                                                                 #
# 1. Redistributions of source code must retain the above copyright notice, this                  #
# list of conditions and the following disclaimer.                                                #
#                                                                                                 #
# 2. Redistributions in binary form must reproduce the above copyright notice, this list          #
# of conditions and the following disclaimer in the documentation and/or other materials          #
# provided with the distribution.                                                                 #
#                                                                                                 #
# 3. Neither the name of the copyright holder nor the names of its contributors may be            #
# used to endorse or promote products derived from this software without specific prior           #
# written permission.                                                                             #
#                                                                                                 #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY             #
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF         #
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL          #
# THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,            #
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF     #
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)          #
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,           #
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS           #
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                                    #
#                                                                                                 #
# Author: Abhijit Bendale (abendale@vast.uccs.edu)                                                #
#                                                                                                 #
# If you use this code, please cite the following works                                           #
#                                                                                                 #
# A. Bendale, T. Boult “Towards Open Set Deep Networks” IEEE Conference on                        #
# Computer Vision and Pattern Recognition (CVPR), 2016                                            #
#                                                                                                 #
# Notice Related to using LibMR.                                                                  #
#                                                                                                 #
# If you use Meta-Recognition Library (LibMR), please note that there is a                        #
# difference license structure for it. The citation for using Meta-Recongition                    #
# library (LibMR) is as follows:                                                                  #
#                                                                                                 #
# Meta-Recognition: The Theory and Practice of Recognition Score Analysis                         #
# Walter J. Scheirer, Anderson Rocha, Ross J. Micheals, and Terrance E. Boult                     #
# IEEE T.PAMI, V. 33, Issue 8, August 2011, pages 1689 - 1695                                     #
#                                                                                                 #
# Meta recognition library is provided with this code for ease of use. However, the actual        #
# link to download latest version of LibMR code is: http://www.metarecognition.com/libmr-license/ #
###################################################################################################


import sys
import scipy.spatial.distance as spd
import numpy as np

try:
    import libmr
except ImportError:
    print("LibMR not installed or libmr.so not found")
    print("Install libmr: cd libMR/; ./compile.sh")
    sys.exit()

# ---------------------------------------------------------------------------------
NCHANNELS = 10

# ---------------------------------------------------------------------------------

labellist = ['postive', 'negative']


def compute_mav_distances(activations, predictions, true_labels):
    """
    Calculates the mean activation vector (MAV) for each class and the distance to the mav for each vector.

    :param activations: logits for each image.
    :param predictions: predicted label for each image.
    :param true_labels: true label for each image.
    :return: MAV and euclidean-cosine distance to each vector.
    """
    correct_activations = list()
    mean_activations = list()
    eucos_dists = list()
    for cl in range(len(np.unique(true_labels))):
        # Find correctly predicted samples and store activation vectors.
        i = (true_labels == predictions)
        i = i & (predictions == cl)
        act = activations[i, :]
        correct_activations.append(act)

        # Compute MAV for class.
        mean_act = np.mean(act, axis=0)
        mean_activations.append(mean_act)

        # Compute all, for this class, correctly classified images' distance to the MAV.
        eucos_dist = list()
        for col in range(len(act)):
            eucos_dist.append(spd.euclidean(mean_act, act[col, :]) / 200. + spd.cosine(mean_act, act[col, :]))
        eucos_dists.append(eucos_dist)
    return mean_activations, eucos_dists


def pgl_weibull_tailfitting(eucos_dist, mean_activations, labellist,
                            tailsize=20,
                            ):
    """ Read through distance files, mean vector and fit weibull model for each category

    Input:
    --------------------------------
    meanfiles_path : contains path to files with pre-computed mean-activation vector
    distancefiles_path : contains path to files with pre-computed distances for images from MAV
    labellist : ImageNet 2012 labellist

    Output:
    --------------------------------
    weibull_model : Perform EVT based analysis using tails of distances and save
                    weibull model parameters for re-adjusting softmax scores    
    """
    print(len(eucos_dist[labellist[-1]]))
    weibull_model = {}
    # for each category, read meanfile, distance file, and perform weibull fitting
    for category in labellist:
        weibull_model[category] = {}
        distance_scores = eucos_dist[category]
        # print('len(distance_scores)',len(distance_scores))
        meantrain_vec = mean_activations[category]
        weibull_model[category]['distances'] = distance_scores
        weibull_model[category]['mean_vec'] = meantrain_vec
        mr = libmr.MR()
        tailtofit = np.sort(distance_scores)[-tailsize:]
        mr.fit_high(tailtofit, len(tailtofit))
        weibull_model[category]['weibull_model'] = mr

    print(weibull_model[labellist[-1]]['weibull_model'])
    return weibull_model


def compute_open_max_probability(openmax_known_score, openmax_unknown_score):
    """
    Compute the OpenMax probability.

    :param openmax_known_score: Weibull scores for known labels.
    :param openmax_unknown_score: Weibull scores for unknown unknowns.
    :return: OpenMax probability.
    """

    prob_closed, prob_open, scores = [], [], []

    # Compute denominator for closet set + open set normalization.
    # Sum up the class scores.
    for category in range(len(openmax_known_score)):
        scores += [np.exp(openmax_known_score[category])]
    total_denominator = np.sum(np.exp(openmax_known_score)) + np.exp(openmax_unknown_score)

    # Scores for image belonging to either closed or open set.
    prob_closed = np.array([scores / total_denominator])
    prob_open = np.array([np.exp(openmax_unknown_score) / total_denominator])

    probs = np.append(prob_closed.tolist(), prob_open)

    # assert len(probs) == 11
    return probs


def recalibrate_scores(weibull_model, img_layer_act, alpharank=3):
    """
    Computes the OpenMax probabilities of an input image.

    :param weibull_model: pre-computed Weibull model.
                          Dictionary with [class_labels]['euclidean distances', 'mean_vec', 'weibull_model']
    :param img_layer_act: activations in penultimate layer.
    :param alpharank: number of top classes to revise/check.
    :return: OpenMax probabilities of image.
    """

    num_labels = alpharank

    # Sort index of activations from highest to lowest.
    ranked_list = np.argsort(img_layer_act)  # 将矩阵a按照axis排序，并返回排序后的下标
    ranked_list = np.ravel(ranked_list)  # 将多维数组降位一维
    ranked_list = ranked_list[::-1]  # 取从后向前（相反）的元素,最终得到从大到小排列的数组

    # Obtain alpha weights for highest -> lowest activations.
    alpha_weights = [((alpharank + 1) - i) / float(alpharank) for i in
                     range(1, alpharank + 1)]  # [10/10,9/10,8/10....1/10]
    ranked_alpha = np.zeros(num_labels)  # (10,)
    for i in range(0, len(alpha_weights)):
        ranked_alpha[ranked_list[i]] = alpha_weights[i]  # 每个索引的权重赋值

    # Calculate OpenMax probabilities
    openmax_penultimate, openmax_penultimate_unknown = [], []
    for categoryid in range(num_labels):
        label_weibull = weibull_model[categoryid]['weibull_model']  # Obtain the corresponding Weibull model.
        label_mav = weibull_model[categoryid]['mean_vec']  # Obtain MAV for specific class.
        img_dist = spd.euclidean(label_mav, img_layer_act) / 200. + spd.cosine(label_mav, img_layer_act)

        weibull_score = label_weibull.w_score(img_dist)

        modified_layer_act = img_layer_act[categoryid] * (1 - weibull_score * ranked_alpha[categoryid])  # Revise av.
        openmax_penultimate += [modified_layer_act]  # Append revised av. to a total list.
        openmax_penultimate_unknown += [img_layer_act[categoryid] - modified_layer_act]  # A.v. 'unknown unknowns'.

    openmax_closedset_logit = np.asarray(openmax_penultimate)
    openmax_openset_logit = np.sum(openmax_penultimate_unknown)

    # Transform the recalibrated penultimate layer scores for the image into OpenMax probability.
    openmax_probab = compute_open_max_probability(openmax_closedset_logit, openmax_openset_logit)

    return openmax_probab


def crosr_recalibrate_scores(weibull_model, img_layer_act, hidden, alpharank=3):
    """
    Computes the OpenMax probabilities of an input image.

    :param weibull_model: pre-computed Weibull model.
                          Dictionary with [class_labels]['euclidean distances', 'mean_vec', 'weibull_model']
    :param img_layer_act: activations in penultimate layer.
    :param alpharank: number of top classes to revise/check.
    :return: OpenMax probabilities of image.
    """

    num_labels = alpharank

    # Sort index of activations from highest to lowest.
    ranked_list = np.argsort(img_layer_act)  # 将矩阵a按照axis排序，并返回排序后的下标
    ranked_list = np.ravel(ranked_list)  # 将多维数组降位一维
    ranked_list = ranked_list[::-1]  # 取从后向前（相反）的元素,最终得到从大到小排列的数组

    # Obtain alpha weights for highest -> lowest activations.
    alpha_weights = [((alpharank + 1) - i) / float(alpharank) for i in
                     range(1, alpharank + 1)]  # [10/10,9/10,8/10....1/10]
    ranked_alpha = np.zeros(num_labels)  # (10,)
    for i in range(0, len(alpha_weights)):
        ranked_alpha[ranked_list[i]] = alpha_weights[i]  # 每个索引的权重赋值

    # Calculate OpenMax probabilities
    openmax_penultimate, openmax_penultimate_unknown = [], []
    for categoryid in range(num_labels):
        label_weibull = weibull_model[categoryid]['weibull_model']  # Obtain the corresponding Weibull model.
        label_mav = weibull_model[categoryid]['mean_vec']  # Obtain MAV for specific class.
        img_dist = spd.euclidean(label_mav, hidden) / 200. + spd.cosine(label_mav, hidden)

        weibull_score = label_weibull.w_score(img_dist)

        modified_layer_act = img_layer_act[categoryid] * (1 - weibull_score * ranked_alpha[categoryid])  # Revise av.
        openmax_penultimate += [modified_layer_act]  # Append revised av. to a total list.
        openmax_penultimate_unknown += [img_layer_act[categoryid] - modified_layer_act]  # A.v. 'unknown unknowns'.

    openmax_closedset_logit = np.asarray(openmax_penultimate)
    openmax_openset_logit = np.sum(openmax_penultimate_unknown)

    # Transform the recalibrated penultimate layer scores for the image into OpenMax probability.
    openmax_probab = compute_open_max_probability(openmax_closedset_logit, openmax_openset_logit)

    return openmax_probab


def save_weibull_model(weibull_model, class_num, MODEL_DIR):
    for category in range(class_num):
        saved = str(weibull_model[category]['weibull_model'])
        array_distances = weibull_model[category]['distances']
        array_mean_vec = weibull_model[category]['mean_vec']
        fh = open(MODEL_DIR + str(category) + 'weibull_model.txt', 'w')
        fh.write(saved)
        fh.close()
        np.save(MODEL_DIR + str(category) + 'array_distances.npy', array_distances)
        np.save(MODEL_DIR + str(category) + 'array_mean_vec.npy', array_mean_vec)


def load_weibull_model(class_num, MODEL_DIR):
    weibull_model = {}
    for category in range(class_num):
        weibull_model[category] = {}
        weibull_model[category]['distances'] = np.load(MODEL_DIR + str(category) + 'array_distances.npy')
        weibull_model[category]['mean_vec'] = np.load(MODEL_DIR + str(category) + 'array_mean_vec.npy')
        with open(MODEL_DIR + str(category) + 'weibull_model.txt') as f:
            read_data = f.read()
        f.closed
        weibull_model[category]['weibull_model'] = libmr.load_from_string(read_data)
    return weibull_model
