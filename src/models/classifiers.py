import pandas as pd
import numpy as np
from rdkit import Chem
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import SVC
import pickle 

df = pd.read_csv('data/interim/jak2_pIC50_data.csv')

def processing(fp_str): 
    fp_lst = [int(i) for i in fp_str]
    fp_np = np.asarray(fp_lst)
    return fp_np

X_train = df[df['dataset_type'] == 'train'].Fingerprint.apply(processing)
X_test = df[df['dataset_type'] == 'test'].Fingerprint.apply(processing)
y_train = df[df['dataset_type'] == 'train'].activity
y_test = df[df['dataset_type'] == 'test'].activity

y_train = y_train.to_numpy()
y_test = y_test.to_numpy()
X_train = list(X_train)
X_test = list(X_test)

def metrics(classifier_name, y_train_predict, y_test_predict, y_train, y_test):
    
    cm_train = confusion_matrix(y_true=y_train_predict, y_pred=y_train)
    cm_test = confusion_matrix(y_true=y_test_predict, y_pred=y_test)
    
    def list_view(array):
        lst = []
        for i in array.tolist():
            for j in i:
                lst.append(j)
        return lst

    cmatrix_train = list_view(cm_train)
    cmatrix_test = list_view(cm_test)
    
    rs_train = recall_score(y_true=y_train_predict, y_pred=y_train)
    rs_test = recall_score(y_true=y_test_predict, y_pred=y_test)
    
    ps_train = precision_score(y_true=y_train_predict, y_pred=y_train)
    ps_test = precision_score(y_true=y_test_predict, y_pred=y_test)
    
    bal_ac_train = balanced_accuracy_score(y_true=y_train_predict, y_pred=y_train)
    bal_ac_test = balanced_accuracy_score(y_true=y_test_predict, y_pred=y_test)
    
    import json
    
    with open('models/metrics.json', 'r') as file:
        data = json.load(file)
        
    data[classifier_name] = {'confusion_matrix_training_set': cmatrix_train,
                             'confusion_matrix_test_set': cmatrix_test,
                             'balanced_accuracy_score_training_set': bal_ac_train,
                             'balanced_accuracy_score_test_set': bal_ac_test,
                             'precision_score_training_set':ps_train,
                             'precision_score_test_set': ps_test,
                             'recall_score_training_set': rs_train,
                             'recall_score_test_set': rs_test
                            }
    
    with open('models/metrics.json', 'w') as file:
        json.dump(data, file, indent=2)
        
        
clf_sgd = make_pipeline(StandardScaler(), SGDClassifier())
clf_sgd.fit(X_train, y_train)

y_train_predict_sgd = clf_sgd.predict(X_train)
y_test_predict_sgd = clf_sgd.predict(X_test)
        
metrics_clf_sgd = metrics('SGDClassifier', y_train_predict_sgd, y_test_predict_sgd, y_train, y_test)

with open("models/SGDClassifier.pkl", 'wb') as file: 
    pickle.dump(clf_sgd, file) 

clf_svc = make_pipeline(StandardScaler(), SVC())
clf_svc.fit(X_train, y_train)

y_train_predict_svc = clf_svc.predict(X_train)
y_test_predict_svc = clf_svc.predict(X_test)

metrics_clf_sgd = metrics('SVC', y_train_predict_svc, y_test_predict_svc, y_train, y_test)

with open("models/SVC.pkl", 'wb') as file: 
    pickle.dump(clf_svc, file) 