import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pickle 
import json

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

def get_confusion_matrix(y_true, y_pred):
        conf_matrix = confusion_matrix(y_true= y_true, y_pred= y_pred)
        return list(map(int, [conf_matrix[0][0], conf_matrix[0][1], conf_matrix[1][0], conf_matrix[1][1]]))
    
def train(X_train, y_train):
    clf = RandomForestClassifier(max_depth=15)
    clf.fit(X_train, y_train)
    return clf
    
def metrics(classifier_name, clf, X_train, X_test, y_train, y_test):

    y_train_predict = clf.predict(X_train)
    y_test_predict = clf.predict(X_test)

    with open('models/metrics.json', 'r') as file:
        data = json.load(file)
        
    data[classifier_name] = {'confusion_matrix_training_set': get_confusion_matrix(y_train, y_train_predict),
                             'confusion_matrix_test_set': get_confusion_matrix(y_test, y_test_predict),
                             'balanced_accuracy_score_training_set': balanced_accuracy_score(y_true=y_train,y_pred=y_train_predict),
                             'balanced_accuracy_score_test_set': balanced_accuracy_score(y_true=y_test, y_pred=y_test_predict),
                             'precision_score_training_set':precision_score(y_true=y_train, y_pred=y_train_predict),
                             'precision_score_test_set': precision_score(y_true=y_test, y_pred=y_test_predict),
                             'recall_score_training_set': recall_score(y_true=y_train, y_pred=y_train_predict),
                             'recall_score_test_set': recall_score(y_true=y_test, y_pred=y_test_predict)
                             }
    
    with open('models/metrics.json', 'w') as file:
        json.dump(data, file, indent=2)
        
def save_model(clf, file_name):
    with open(f'models/{file_name}', 'wb') as file: 
         pickle.dump(clf , file) 
 
def main():
    rf_clf = train(X_train, y_train)
    metrics('RandomForestClassifier', rf_clf, X_train, X_test, y_train, y_test)
    save_model(rf_clf, 'RFClassifier.pkl')

if __name__ == '__main__':
    main()
