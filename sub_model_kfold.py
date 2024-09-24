import numpy as np
import pandas as pd
import os
from PC6_encoding import get_PC6_features_labels
from doc2vec import get_Doc2Vec_features_labels
from sklearn import metrics
from model_tools import learning_curve, evalution_metrics, findThresIndex
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import joblib

pos_train_data = './data/positive_2638.fasta'
neg_train_data = './data/negative_2633.fasta'

# Encoding through pc6 pretrained
pc6_train_features, pc6_train_labels = get_PC6_features_labels(pos_train_data, neg_train_data,length=1024)
reshape_pc6_train_features = pc6_train_features.reshape(pc6_train_features.shape[0],-1)

# Encoding through Doc2Vec pretrained
#doc2vec_model = './Doc2Vec_model/AFP_doc2vec.model'
doc2vec_train_features, doc2vec_train_labels = get_Doc2Vec_features_labels(pos_train_data, neg_train_data, './Doc2Vec_model/surfaceome_doc2vec.model')
reshape_doc2vec_train_features=doc2vec_train_features.reshape((doc2vec_train_features.shape[0],doc2vec_train_features.shape[1],1))

print("PC6 train encoding:")
print(pc6_train_features.shape)
print(reshape_pc6_train_features.shape)

print("Doc2vec train encoding:")
print(doc2vec_train_features.shape)
#print(reshape_doc2vec_train_features.shape)

from sklearn import ensemble
from sklearn import svm
from model import train_pc6_model
from model import train_doc2vec_model
from sklearn.model_selection import KFold

# Create directory
if not os.path.isdir('ensemble_10_fold'):
    os.makedirs('ensemble_10_fold/pc6')
    os.makedirs('ensemble_10_fold)/doc2vec')

def fold_cv(train_data, labels, mode, output_dir = '.'):
    score_array = []
    label_array = []
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    # K-fold Cross Validation model evaluation
    df = pd.DataFrame(columns=['TP', 'TN', 'FP', 'FN', 'accuracy', 'precision', 'sensitivity', 'specificity', 'f1', 'mcc'])
    fold_no = 1
    print(train_data)
    print(labels)
    for train, val in kfold.split(train_data, labels):
        # Generate a print
        print('------------------------------------------------------------------------')
        print('Training for fold:')
        print(fold_no)
        if mode == 'svm':
            svc = svm.SVC()
            svc_fit = svc.fit(train_data[train], labels[train])
            labels_score = svc.predict(train_data[val])
            joblib.dump(svc, os.path.join(output_dir, 'svm_%s.pkl'%fold_no))
            label_array.append(labels[val])
        if mode == 'rf':
            forest = ensemble.RandomForestClassifier(n_estimators = 100)
            forest_fit = forest.fit(train_data[train], labels[train])
            labels_score = forest.predict(train_data[val])
            joblib.dump(forest, os.path.join(output_dir, 'forest_%s.pkl'%fold_no))
            label_array.append(labels[val])
        if mode == 'pc6_cnn':
            pc6_path = 'pc6_threshold.txt'
            pc6_f = open(pc6_path, 'a')
            labels_score = []
            train_pc6_model(train_data[train], labels[train], train_data[val], labels[val], model_name = 'kfold%s'%fold_no, path = output_dir)
            model = load_model(os.path.join(output_dir, 'kfold%s_best_weights.h5'%fold_no))
            temp_labels_score = model.predict(train_data[val])
            (pc6_fpr, pc6_tpr, pc6_thresholds) = metrics.roc_curve(labels[val], temp_labels_score)

            pc6_thresidx = findThresIndex(pc6_tpr, pc6_fpr)
            pc6_thres = pc6_thresholds[pc6_thresidx]
            if type(pc6_thres) is np.ndarray:
                pc6_thres = 0.5
            pc6_f.write(str(fold_no) + '\t')
            pc6_f.write(str(pc6_thres) + '\n')
            for i in range(len(labels[val])):
                if(temp_labels_score[i][0] >= pc6_thres):
                    labels_score.append(float('1'))
                else:
                    labels_score.append(float('0'))
            label_array.append(labels[val])
            pc6_f.close()
        if mode == 'd2v_cnn':
            d2v_path = 'd2v_threshold.txt'
            d2v_f = open(d2v_path, 'a')
            labels_score = []
            train_doc2vec_model(train_data[train], labels[train], train_data[val], labels[val], model_name = 'kfold%s'%fold_no, path = output_dir)
            model = load_model(os.path.join(output_dir, 'kfold%s_best_weights.h5'%fold_no))
            temp_labels_score = model.predict(train_data[val])
            (doc2vec_fpr, doc2vec_tpr, doc2vec_thresholds) = metrics.roc_curve(labels[val], temp_labels_score)

            doc2vec_thresidx = findThresIndex(doc2vec_tpr, doc2vec_fpr)
            doc2vec_thres = doc2vec_thresholds[doc2vec_thresidx]
            if type(doc2vec_thres) is np.ndarray:
                doc2vec_thres = 0.5
            d2v_f.write(str(fold_no) + '\t')
            d2v_f.write(str(doc2vec_thres) + '\n')
            for i in range(len(labels[val])):
                if(temp_labels_score[i][0] >= doc2vec_thres):
                    labels_score.append(float('1'))
                else:
                    labels_score.append(float('0'))
            label_array.append(labels[val])
            d2v_f.close()
        score_array.append(labels_score)
        metrics_dict = evalution_metrics(labels[val], np.array(labels_score), save=False)
        #print(metrics_dict)
        df.loc[fold_no] = metrics_dict.values()
        # Increase fold number
        fold_no = fold_no + 1
    df.loc['Mean'] = df.mean()
    df.to_csv(os.path.join(output_dir,'%s_cv.csv'%mode))
    #return(df)
    return(score_array, label_array)

# Run&Write PC6 results
pc6_svm_res, pc6_svm_label = fold_cv(reshape_pc6_train_features, pc6_train_labels, mode='svm', output_dir = './ensemble_10_fold/pc6')
with open('./ensemble_10_fold/pc6_svm_res.txt', 'w') as fp:
    for item in pc6_svm_res:
        fp.write("%s\n" % item)
pc6_rf_res, pc6_rf_label = fold_cv(reshape_pc6_train_features, pc6_train_labels, mode='rf', output_dir = './ensemble_10_fold/pc6')
with open('./ensemble_10_fold/pc6_rf_res.txt', 'w') as fp:
    for item in pc6_rf_res:
        fp.write("%s\n" % item)
pc6_cnn_res, pc6_cnn_label = fold_cv(pc6_train_features, pc6_train_labels, mode='pc6_cnn', output_dir = './ensemble_10_fold/pc6')
with open('./ensemble_10_fold/pc6_cnn_res.txt', 'w') as fp:
    for item in pc6_cnn_res:
        fp.write("%s\n" % item)

# Run&Write Doc2vec results
d2v_svm_res, d2v_svm_label = fold_cv(doc2vec_train_features, doc2vec_train_labels, mode='svm', output_dir = './ensemble_10_fold/doc2vec')
with open('./ensemble_10_fold/d2v_svm_res.txt', 'w') as fp:
    for item in d2v_svm_res:
        fp.write("%s\n" % item)
d2v_rf_res, d2v_rf_label = fold_cv(doc2vec_train_features, doc2vec_train_labels, mode='rf', output_dir = './ensemble_10_fold/doc2vec')
with open('./ensemble_10_fold/d2v_rf_res.txt', 'w') as fp:
    for item in d2v_rf_res:
        fp.write("%s\n" % item)
d2v_cnn_res, d2v_cnn_label = fold_cv(reshape_doc2vec_train_features, doc2vec_train_labels, mode='d2v_cnn', output_dir = './ensemble_10_fold/doc2vec')
with open('./ensemble_10_fold/d2v_cnn_res.txt', 'w') as fp:
    for item in d2v_cnn_res:
        fp.write("%s\n" % item)

# Run ensemble model 10fold
from model import train_ensemble_model
ensemble_len = len(pc6_svm_res)
for i in range(0, ensemble_len):
    ensembleX_list = []
    ensembleY_list = []
    
    pc6_svm = pc6_svm_res[i]
    pc6_rf = pc6_rf_res[i]
    pc6_cnn = pc6_cnn_res[i]
    
    d2v_svm = d2v_svm_res[i]
    d2v_rf = d2v_rf_res[i]
    d2v_cnn = d2v_cnn_res[i]
    
    fold_len = len(pc6_svm)
    for j in range(0, fold_len):
        score_list = []
        score_list.append(float(pc6_svm[j]))
        score_list.append(float(pc6_rf[j]))
        score_list.append(float(pc6_cnn[j]))
        score_list.append(float(d2v_svm[j]))
        score_list.append(float(d2v_rf[j]))
        score_list.append(float(d2v_cnn[j]))
        
        ensembleX_list.append(score_list)
    ensembleX = np.array(ensembleX_list)
    ensembleY = pc6_svm_label[i]
    fold = str(i+1)
    e_m = train_ensemble_model(ensembleX, ensembleY, '10fold_ensemble_%s'%fold, path = './ensemble_10_fold')

fp.close()

