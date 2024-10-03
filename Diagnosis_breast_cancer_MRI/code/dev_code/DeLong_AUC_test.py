#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 12:16:27 2023

@author: deeperthought
"""

import os
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import scipy.stats


import sklearn.datasets
import sklearn.model_selection
import sklearn.linear_model    
import scipy.stats

# AUC comparison adapted from
# https://github.com/Netflix/vmaf/
def compute_midrank(x):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5*(i + j - 1)
        i = j
    T2 = np.empty(N, dtype=np.float)
    # Note(kazeevn) +1 is due to Python using 0-based indexing
    # instead of 1-based in the AUC formula in the paper
    T2[J] = T + 1
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Operating Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float)
    ty = np.empty([k, n], dtype=np.float)
    tz = np.empty([k, m + n], dtype=np.float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def calc_pvalue(aucs, sigma):
    """Computes log(10) of p-values.
    Args:import os
    import numpy as np
    import nibabel as nib
    import pandas as pd
    import matplotlib.pyplot as plt

    import pandas as pd
    import numpy as np
    import scipy.stats


    import sklearn.datasets
    import sklearn.model_selection
    import sklearn.linear_model    
    import scipy.stats
       aucs: 1D array of AUCs
       sigma: AUC DeLong covariances
    Returns:get_DeLong_pValue
       log10(pvalue)
    """
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    return np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1) / np.log(10), z


def compute_ground_truth_statistics(ground_truth):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    return order, label_1_count


def delong_roc_variance(ground_truth, predictions):
    """
    Computes ROC AUC variance for a single set of predictions
    Args:
       ground_truth: np.array of 0 and 1
       predictions: np.array of floats of the probability of being class 1
    """
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"
    return aucs[0], delongcov


def delong_roc_test(ground_truth, predictions_one, predictions_two):
    """
    Computes log(p-value) for hypothesis that two ROC AUCs are different
    Args:
       ground_truth: np.array of 0 and 1
       predictions_one: predictions of the first model,
          np.array of floats of the probability of being class 1
       predictions_two: predictions of the second model,
          np.array of floats of the probability of being class 1
    """
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = np.vstack((predictions_one, predictions_two))[:, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    return calc_pvalue(aucs, delongcov)


        
#%%
#    
#x_distr = scipy.stats.norm(0.5, 1)
#y_distr = scipy.stats.norm(-0.5, 1)
#sample_size_x = 7
#sample_size_y = 14
#n_trials = 1000
#aucs = np.empty(n_trials)
#variances = np.empty(n_trials)
#np.random.seed(1234235)
#labels = np.concatenate([np.ones(sample_size_x), np.zeros(sample_size_y)])
#for trial in range(n_trials):
#    scores = np.concatenate([
#        x_distr.rvs(sample_size_x),
#        y_distr.rvs(sample_size_y)])
#    aucs[trial] = sklearn.metrics.roc_auc_score(labels, scores)
#    auc_delong, variances[trial] = delong_roc_variance(
#        labels, scores)
#
#print("Experimental variance {}, computed vairance {}, {} trials".format(variances.mean(),aucs.var() ,n_trials))
#
#
#
##%% Compare 2 models
#
#
#CNN_SMALL_DATA_01_PATH = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/SmallerData0.1_classifier_train9833_val5892_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/VAL_result.csv'
#CNN_SMALL_DATA_05_PATH = '/home/deeperthought/Projects/DGNS/2D_Diagnosis_model/Sessions/PAPER_SESSIONS/Half_of_data_for_training_classifier_train39064_val5892_DataAug_Clinical_depth6_filters42_L21e-05_batchsize8/VAL_result.csv'


RESULTS_TABLE_PATH1 = '/home/deeperthought/Documents/Papers_and_grants/Risk_paper/data/RISK/results_sheet_BIRADS123.csv'#'/home/deeperthought/Projects/DGNS/Risk_Prediction/Sessions/FINAL/training_dropout0.5_classWeight1.0_Clinical/paper_selected/class_weight1/results_sheet.csv'

RESULTS_TABLE_PATH2 = '/home/deeperthought/Projects/DGNS/Risk_Prediction/Sessions/FINAL/training_dropout0.5_classWeight1.0_Clinical/paper_selected/class_weight1/test_results_replication/PAPER_DATA_NoDemographics_GLOBAL_RESULTS_TEST.csv'



CNN_SMALL_DATA_01 = pd.read_csv(RESULTS_TABLE_PATH1)
CNN_SMALL_DATA_05 = pd.read_csv(RESULTS_TABLE_PATH2)

results = pd.merge(CNN_SMALL_DATA_01, CNN_SMALL_DATA_05, on='scanID')

results[['y_true_x','y_true_y']]



p,z = delong_roc_test(results['y_true_x'], results['y_pred_x'],  results['y_pred_y'])


get_DeLong_pValue(results)

#
#
##%%
#
#import sklearn.datasets
#import sklearn.model_selection
#import sklearn.linear_model    
#import numpy
#import compare_auc_delong_xu
#import unittest
#import scipy.stats
#
#class TestIris(unittest.TestCase):
#    @classmethod
#    def setUpClass(cls):
#        data = sklearn.datasets.load_iris()
#        x_train, x_test, y_train, cls.y_test = sklearn.model_selection.train_test_split(
#            data.data, (data.target == 1).astype(numpy.int), test_size=0.8, random_state=42)
#        cls.predictions = sklearn.linear_model.LogisticRegression(solver="lbfgs").fit(
#            x_train, y_train).predict_proba(x_test)[:, 1]
#        cls.sklearn_auc = sklearn.metrics.roc_auc_score(cls.y_test, cls.predictions)
#
#    def test_variance_const(self):
#        auc, variance = compare_auc_delong_xu.delong_roc_variance(self.y_test, self.predictions)
#        numpy.testing.assert_allclose(self.sklearn_auc, auc)
#        numpy.testing.assert_allclose(0.0015359814789736538, variance)
#
#
#class TestGauss(unittest.TestCase):
#    x_distr = scipy.stats.norm(0.5, 1)
#    y_distr = scipy.stats.norm(-0.5, 1)
#
#    def test_variance(self):
#        sample_size_x = 7
#        sample_size_y = 14
#        n_trials = 50000
#        aucs = numpy.empty(n_trials)
#        variances = numpy.empty(n_trials)
#        numpy.random.seed(1234235)
#        labels = numpy.concatenate([numpy.ones(sample_size_x), numpy.zeros(sample_size_y)])
#        for trial in range(n_trials):
#            scores = numpy.concatenate([
#                self.x_distr.rvs(sample_size_x),
#                self.y_distr.rvs(sample_size_y)])
#            aucs[trial] = sklearn.metrics.roc_auc_score(labels, scores)
#            auc_delong, variances[trial] = compare_auc_delong_xu.delong_roc_variance(
#                labels, scores)
#            numpy.testing.assert_allclose(aucs[trial], auc_delong)
#        numpy.testing.assert_allclose(variances.mean(), aucs.var(), rtol=0.1)
