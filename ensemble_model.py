#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
from PC6_encoding import get_PC6_features_labels
from doc2vec import get_Doc2Vec_features_labels
#from model import train_pc6_model
from model_tools import learning_curve, evalution_metrics
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import joblib

# Load data
pos_train_data = './data/pos_ensemble_trainset_2110.fasta'
neg_train_data = './data/neg_ensemble_trainset_2110.fasta'
pos_valid_data = './data/pos_ensemble_validset_264.fasta'
neg_valid_data = './data/neg_ensemble_validset_264.fasta'
pos_test_data = './data/pos_testset_264.fasta'
neg_test_data = './data/neg_testset_259.fasta'

# Encoding through pc6 pretrained 
pc6_train_features, pc6_train_labels = get_PC6_features_labels(pos_train_data, neg_train_data,length=1024)
reshape_pc6_train_features = pc6_train_features.reshape(pc6_train_features.shape[0],-1)

pc6_valid_features, pc6_valid_labels = get_PC6_features_labels(pos_valid_data, neg_valid_data,length=1024)
reshape_pc6_valid_features = pc6_valid_features.reshape(pc6_valid_features.shape[0],-1)

pc6_test_features, pc6_test_labels = get_PC6_features_labels(pos_test_data, neg_test_data,length=1024)
reshape_pc6_test_features = pc6_test_features.reshape(pc6_test_features.shape[0],-1)

# Encoding through Doc2Vec pretrained
#doc2vec_model = './Doc2Vec_model/AFP_doc2vec.model'
doc2vec_train_features, doc2vec_train_labels = get_Doc2Vec_features_labels(pos_train_data, neg_train_data, './Doc2Vec_model/surfaceome_doc2vec.model')
reshape_doc2vec_train_features=doc2vec_train_features.reshape((doc2vec_train_features.shape[0],doc2vec_train_features.shape[1],1))

doc2vec_valid_features, doc2vec_valid_labels = get_Doc2Vec_features_labels(pos_valid_data, neg_valid_data, './Doc2Vec_model/surfaceome_doc2vec.model')
reshape_doc2vec_valid_features=doc2vec_valid_features.reshape((doc2vec_valid_features.shape[0],doc2vec_valid_features.shape[1]))

doc2vec_test_features, doc2vec_test_labels = get_Doc2Vec_features_labels(pos_test_data, neg_test_data, './Doc2Vec_model/surfaceome_doc2vec.model')
reshape_doc2vec_test_features=doc2vec_test_features.reshape((doc2vec_test_features.shape[0],doc2vec_test_features.shape[1]))

# check labels
print(pc6_train_labels)
print(doc2vec_train_labels)
train_labels = np.array(pc6_train_labels)
valid_labels = np.array(pc6_valid_labels)
test_labels = np.array(pc6_test_labels)

from sklearn import ensemble
from sklearn import svm
from model import train_pc6_model
from model import train_doc2vec_model
from bert_prediction import get_bert_prediction, train_bert_model, run_bert_prediction

#PC6 model
pc6path = './ensemble_model/pc6' 
if not os.path.exists(pc6path):
    os.makedirs(pc6path)
#RF
pc6_forest = ensemble.RandomForestClassifier(n_estimators = 100)
pc6_forest_fit = pc6_forest.fit(reshape_pc6_train_features, pc6_train_labels)
joblib.dump(pc6_forest, './ensemble_model/pc6/pc6_features_forest.pkl')

pc6_forest = joblib.load('./ensemble_model/pc6/pc6_features_forest.pkl')
pc6_rf_labels_score = pc6_forest.predict(reshape_pc6_valid_features)
pc6_rf_res = evalution_metrics(pc6_valid_labels, pc6_rf_labels_score)

#SVM
pc6_svc = svm.SVC()
pc6_svc_fit = pc6_svc.fit(reshape_pc6_train_features, pc6_train_labels)
joblib.dump(pc6_svc, './ensemble_model/pc6/pc6_features_svm.pkl')

pc6_svc = joblib.load('./ensemble_model/pc6/pc6_features_svm.pkl')
pc6_svm_labels_score = pc6_svc.predict(reshape_pc6_valid_features)
pc6_svm_res = evalution_metrics(pc6_valid_labels, pc6_svm_labels_score)

#CNN
pc6_train_data_, pc6_test_data_, pc6_train_labels_, pc6_test_labels_ = train_test_split(pc6_train_features, pc6_train_labels, test_size= 0.1, random_state = 1, stratify = pc6_train_labels)
pc6_t_m = train_pc6_model(pc6_train_data_, pc6_train_labels_, pc6_test_data_, pc6_test_labels_, 'ensemble', path = './ensemble_model/pc6')

#learning_curve(pc6_t_m.history)
learning_curve(pc6_t_m.history, save = True, output_path_name = './ensemble_model/pc6')

pc6_cnn_model = load_model('./ensemble_model/pc6/ensemble_best_weights.h5')
pc6_cnn_labels_score = pc6_cnn_model.predict(pc6_valid_features)
pc6_cnn_res = evalution_metrics(pc6_valid_labels, pc6_cnn_labels_score)

#doc2vec model
d2vpath = './ensemble_model/doc2vec' 
if not os.path.exists(d2vpath):
    os.makedirs(d2vpath)
#RF
doc2vec_forest = ensemble.RandomForestClassifier(n_estimators = 100)
doc2vec_forest_fit = doc2vec_forest.fit(doc2vec_train_features, doc2vec_train_labels)
joblib.dump(doc2vec_forest, './ensemble_model/doc2vec/doc2vec_features_forest.pkl')

doc2vec_forest = joblib.load('./ensemble_model/doc2vec/doc2vec_features_forest.pkl')
doc2vec_rf_labels_score = doc2vec_forest.predict(doc2vec_valid_features)
doc2vec_rf_res = evalution_metrics(doc2vec_valid_labels, doc2vec_rf_labels_score)

#SVM
doc2vec_svc = svm.SVC()
doc2vec_svc_fit = doc2vec_svc.fit(doc2vec_train_features, doc2vec_train_labels)
joblib.dump(doc2vec_svc, './ensemble_model/doc2vec/doc2vec_features_svm.pkl')

doc2vec_svc = joblib.load('./ensemble_model/doc2vec/doc2vec_features_svm.pkl')
doc2vec_svm_labels_score = doc2vec_svc.predict(doc2vec_valid_features)
doc2vec_svm_res = evalution_metrics(doc2vec_valid_labels, doc2vec_svm_labels_score)

#CNN
doc2vec_train_data_, doc2vec_test_data_, doc2vec_train_labels_, doc2vec_test_labels_ = train_test_split(reshape_doc2vec_train_features, doc2vec_train_labels, test_size= 0.1, random_state = 1, stratify = pc6_train_labels)
doc2vec_t_m = train_doc2vec_model(doc2vec_train_data_, doc2vec_train_labels_, doc2vec_test_data_, doc2vec_test_labels_, 'ensemble', path = './ensemble_model/doc2vec')

#learning_curve(doc2vec_t_m.history)
learning_curve(doc2vec_t_m.history, save = True, output_path_name = './ensemble_model/doc2vec')

doc2vec_cnn_model = load_model('./ensemble_model/doc2vec/ensemble_best_weights.h5')
doc2vec_cnn_labels_score = doc2vec_cnn_model.predict(reshape_doc2vec_valid_features)
doc2vec_cnn_res = evalution_metrics(doc2vec_valid_labels, doc2vec_cnn_labels_score)
#print(doc2vec_cnn_labels_score)

# Find model threshold
from model_tools import evalution_metrics, findThresIndex
from sklearn import metrics
(pc6_fpr, pc6_tpr, pc6_thresholds) = metrics.roc_curve(pc6_valid_labels, pc6_cnn_labels_score)
(doc2vec_fpr, doc2vec_tpr, doc2vec_thresholds) = metrics.roc_curve(doc2vec_valid_labels, doc2vec_cnn_labels_score)

pc6_thresidx = findThresIndex(pc6_tpr, pc6_fpr)
pc6_thres = pc6_thresholds[pc6_thresidx]

doc2vec_thresidx = findThresIndex(doc2vec_tpr, doc2vec_fpr)
doc2vec_thres = doc2vec_thresholds[doc2vec_thresidx]
print('PC6_CNN_thres=' + str(pc6_thres))
print('Doc2vec_CNN_thres=' +  str(doc2vec_thres))

#Generate ensemble model input
from model import train_ensemble_model
valid_size = len(pc6_valid_labels)

ensembleX_list = []
ensembleY_list = []
for i in range(valid_size):
    score_list = []
    score_list.append(float(pc6_rf_labels_score[i]))
    score_list.append(float(pc6_svm_labels_score[i]))
    if(pc6_cnn_labels_score[i][0] >= pc6_thres):
        score_list.append(float('1'))
    else:
        score_list.append(float('0'))
    #score_list.append(float(PC6_nn_labels_score[i][0]))

    score_list.append(float(doc2vec_rf_labels_score[i]))
    score_list.append(float(doc2vec_svm_labels_score[i]))
    if(doc2vec_cnn_labels_score[i][0] >= doc2vec_thres):
        score_list.append(float('1'))
    else:
        score_list.append(float('0'))
    #score_list.append(float(Doc2vec_nn_labels_score[i][0]))
    ensembleX_list.append(score_list)

#print(len(ensembleX_list))
#print(len(ensembleY_list))
ensembleX = np.array(ensembleX_list)
ensembleY = np.array(pc6_valid_labels)

e_m = train_ensemble_model(ensembleX, ensembleY, 'ensemble', path = './ensemble_model')
 
#print(pc6_rf_labels_score.shape)
#print(pc6_svm_labels_score.shape)
#pc6_cnn_labels_score = np.reshape(pc6_cnn_labels_score,(np.size(pc6_cnn_labels_score), ))
#print(pc6_cnn_labels_score.shape)
#print(doc2vec_rf_labels_score.shape)
#print(doc2vec_svm_labels_score.shape)
#doc2vec_cnn_labels_score = np.reshape(doc2vec_cnn_labels_score,(np.size(doc2vec_cnn_labels_score), ))
#print(doc2vec_cnn_labels_score.shape)
#bert_labels=np.array(bert_labels)
#print(bert_labels.shape)

#merge_valid_score = np.stack((pc6_rf_labels_score, pc6_svm_labels_score, pc6_cnn_labels_score, doc2vec_rf_labels_score, doc2vec_svm_labels_score, doc2vec_cnn_labels_score, bert_labels), axis=1)
#print(merge_valid_score)

# Get independent data size
in_size = len(pc6_test_labels)

pc6_rf_test_score = pc6_forest.predict(reshape_pc6_test_features)
pc6_svm_test_score = pc6_svc.predict(reshape_pc6_test_features)
pc6_cnn_test_score = pc6_cnn_model.predict(pc6_test_features)

doc2vec_rf_test_score = doc2vec_forest.predict(doc2vec_test_features)
doc2vec_svm_test_score = doc2vec_svc.predict(doc2vec_test_features)
doc2vec_cnn_test_score = doc2vec_cnn_model.predict(reshape_doc2vec_test_features)

in_ensembleX_list = []
in_ensembleY_list = []
for i in range(in_size):
    #print(Doc2vec_rf_labels_score[i])
    score_list = []
    score_list.append(float(pc6_rf_test_score[i]))
    score_list.append(float(pc6_svm_test_score[i]))
    if(pc6_cnn_test_score[i][0] >= pc6_thres):
        score_list.append(float('1'))
    else:
        score_list.append(float('0'))
    #score_list.append(float(PC6_nn_in_score[i][0]))

    score_list.append(float(doc2vec_rf_test_score[i]))
    score_list.append(float(doc2vec_svm_test_score[i]))
    if(doc2vec_cnn_test_score[i][0] >= doc2vec_thres):
        score_list.append(float('1'))
    else:
        score_list.append(float('0'))
        
    in_ensembleX_list.append(score_list)

#print(len(ensembleX_list))
#print(len(ensembleY_list))
#print(in_ensembleX_list)
in_ensembleX = np.array(in_ensembleX_list)
in_ensembleY = np.array(pc6_test_labels)

ensemble_model = load_model('./ensemble_model/ensemble_best_weights.h5')
in_score = ensemble_model.predict(in_ensembleX)
#print(in_score.ravel().tolist())
#print(in_ensembleY)
final_score = evalution_metrics(pc6_test_labels, in_score)
print("Final score:")
print(final_score)
print("PC6 RF score:")
print(pc6_rf_res)
print("PC6 SVM score:")
print(pc6_svm_res)
print("PC6 CNN score:")
print(pc6_cnn_res)
print("Doc2vec RF score:")
print(doc2vec_rf_res)
print("Doc2vec SVM score:")
print(doc2vec_svm_res)
print("Doc2vec CNN score:")
print(doc2vec_cnn_res)
