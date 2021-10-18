import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.linear_model import SGDClassifier, RidgeClassifier, LogisticRegression
from sklearn.metrics import classification_report, make_scorer, f1_score, accuracy_score, precision_score, \
    recall_score, roc_auc_score, multilabel_confusion_matrix, matthews_corrcoef
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
import itertools


# Used in Task 1 to convert class into one hot encoded vectors
def one_hot_encoding(labels):
    dictionary = {1: [0, 0, 1],
                  2: [0, 1, 0],
                  3: [1, 0, 0]}
    enc_labels = []
    for el in labels:
        enc_labels.append(dictionary[el])
    return np.array(enc_labels)


def define_scores():
    scores = [accuracy_score, precision_score, recall_score, f1_score]
    score_names = ['accuracy', 'precision', 'recall', 'f1']
    made_scores = [make_scorer(score, average='weighted', zero_division=0, pos_label=1)
                   if score != accuracy_score else make_scorer(score) for score in scores]
    metrics = dict(zip(score_names, made_scores))
    return metrics


# MCC function that compute the score for every pair of truth-predicted labels and then compute the mean
def computeMCC(y_test, y_pred):
    value = 0
    for y1, y2 in zip(y_test, y_pred):
        '''if sum(y1) == 0:
            y1 = np.append(y1, 1)
        else:
            y1 = np.append(y1, 0)
        if sum(y2) == 0:
            y2 = np.append(y2, 1)
        else:
            y2 = np.append(y2, 0)'''
        try:
            value += matthews_corrcoef(y1, y2)
        except ValueError:
            print(y1)
            print(y2)
            exit(1)
    mcc = value / len(y_test)
    return mcc


# MCC function that compute score for each label
def computeMCCclass(y_test, y_pred):
    mccs = []
    for i in range(len(y_test[0,:])):
        mccs.append(matthews_corrcoef(y_test[:,i], y_pred[:,i]))
    return mccs


def apply_fold(X_train, X_test, y_train, y_test, clf, params, metrics, task):
    if task == 1:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    else:
        skf = KFold(n_splits=5, shuffle=True, random_state=42)
    gridsearch = GridSearchCV(clf, param_grid=params, scoring=metrics, cv=skf, refit='f1')
    gridsearch.fit(X_train, y_train)
    best_parameters = gridsearch.best_params_
    print(f'Best parameters: {best_parameters}')
    model = gridsearch.best_estimator_
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    result = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    # result = []
    if task == 1:
        en_y_pred = one_hot_encoding(y_pred)
        en_y_test = one_hot_encoding(y_test)
    else:
        en_y_pred = y_pred
        en_y_test = y_test
    roc = roc_auc_score(en_y_test, en_y_pred, average='weighted', multi_class='ovr')
    if task == 1:
        mcc = matthews_corrcoef(y_test, y_pred)
    else:
        mcc = computeMCCclass(np.array(en_y_test), np.array(en_y_pred))
    return result, roc, mcc


def method_applier(X, y, clf_f, metrics, task):
    if task == 1:
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=0)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    result, roc, mcc = clf_f(X_train, X_test, y_train, y_test, metrics, task)
    return result, roc, mcc


def GaussianNB_Classifier(X_train, X_test, y_train, y_test, metrics, task):
    if task == 2:
        clf = MultiOutputClassifier(GaussianNB())
        param_grid = {}
    else:
        clf = GaussianNB()
        param_grid = {}
    cm, report, auc = apply_fold(X_train.toarray(), X_test.toarray(), y_train, y_test, clf, param_grid, metrics, task)
    # print(f'Best parameters: {clf.best_params_}')
    return cm, report, auc


def BernoulliNB_Classifier(X_train, X_test, y_train, y_test, metrics, task):
    if task == 2:
        clf = MultiOutputClassifier(BernoulliNB())
        param_grid = {'estimator__alpha': [0.001, 0.01, 0.1, 1, 10]}
        #param_grid = {}
    else:
        clf = BernoulliNB()
        param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10]}
    cm, report, auc = apply_fold(X_train.toarray(), X_test.toarray(), y_train, y_test, clf, param_grid, metrics, task)
    # print(f'Best parameters: {clf.best_params_}')
    return cm, report, auc


def DecisionTree_Classifier(X_train, X_test, y_train, y_test, metrics, task):
    if task == 2:
        clf = MultiOutputClassifier(DecisionTreeClassifier())
        param_grid = {
            'estimator__criterion': ["gini", "entropy"],
            'estimator__splitter': ["best", "random"],
        }
    else:
        clf = DecisionTreeClassifier()
        param_grid = {
            'criterion': ["gini", "entropy"],
            'splitter': ["best", "random"],
        }
    cm, report, auc = apply_fold(X_train, X_test, y_train, y_test, clf, param_grid, metrics, task)
    # print(f'Best parameters: {clf.best_params_}')
    return cm, report, auc


def RandomForest_Classifier(X_train, X_test, y_train, y_test, metrics, task):
    if task == 2:
        clf = MultiOutputClassifier(RandomForestClassifier())
        param_grid = {'estimator__n_estimators': [100, 200, 250, 300]}
    else:
        clf = RandomForestClassifier()
        param_grid = {'n_estimators': [100, 200, 250, 300]}
    cm, report, auc = apply_fold(X_train, X_test, y_train, y_test, clf, param_grid, metrics, task)
    # print(f'Best parameters: {clf.best_params_}')
    return cm, report, auc


def SVC_Classifier(X_train, X_test, y_train, y_test, metrics, task):
    if task == 2:
        clf = MultiOutputClassifier(SVC())
        param_grid = {'estimator__C': [1, 2, 3],
                      'estimator__kernel': ['linear', 'rbf', 'sigmoid'],
                      'estimator__gamma': ['scale', 'auto']}
    else:
        clf = SVC()
        param_grid = {'C': [1, 2, 3],
                      'kernel': ['linear', 'rbf', 'sigmoid'],
                      'gamma': ['scale', 'auto']}
    cm, report, auc = apply_fold(X_train, X_test, y_train, y_test, clf, param_grid, metrics, task)
    return cm, report, auc


def SGD_Classifier(X_train, X_test, y_train, y_test, metrics, task):
    if task == 2:
        clf = MultiOutputClassifier(SGDClassifier(max_iter=1000))
        param_grid = {'estimator__loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
                      'estimator__penalty': ['l2', 'l1', 'elasticnet'],
                      'estimator__alpha': [1e-4, 1e-3, 1e-2, 1e-1]}
    else:
        clf = SGDClassifier(max_iter=1000)
        param_grid = {'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
                      'penalty': ['l2', 'l1', 'elasticnet'],
                      'alpha': [1e-4, 1e-3, 1e-2, 1e-1]}
    cm, report, auc = apply_fold(X_train, X_test, y_train, y_test, clf, param_grid, metrics, task)
    # print(f'Best parameters: {clf.best_params_}')
    return cm, report, auc


def AdaBoost_Classifier(X_train, X_test, y_train, y_test, metrics, task):
    if task == 2:
        clf = MultiOutputClassifier(AdaBoostClassifier())
        param_grid = {'estimator__n_estimators': [120, 20, 50, 100, 500],
                      'estimator__learning_rate': [0.01, 0.1, 0.5, 1, 10]}
    else:
        clf = AdaBoostClassifier()
        param_grid = {'n_estimators': [120, 20, 50, 100, 500],
                      'learning_rate': [0.01, 0.1, 0.5, 1, 10]}
    cm, report, auc = apply_fold(X_train, X_test, y_train, y_test, clf, param_grid, metrics, task)
    # print(f'Best parameters: {clf.best_params_}')
    return cm, report, auc


def Ridge_Classifier(X_train, X_test, y_train, y_test, metrics, task):
    if task == 2:
        clf = MultiOutputClassifier(RidgeClassifier())
        param_grid = {'estimator__alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1, 10],
                      'estimator__fit_intercept': [True, False],
                      'estimator__tol': [1e-2, 1e-3, 1e-4]}
    else:
        clf = RidgeClassifier()
        param_grid = {'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1, 10],
                      'fit_intercept': [True, False],
                      'tol': [1e-2, 1e-3, 1e-4]}
    cm, report, auc = apply_fold(X_train, X_test, y_train, y_test, clf, param_grid, metrics, task)
    # print(f'Best parameters: {clf.best_params_}')
    return cm, report, auc


def LogisticRegression_Classifier(X_train, X_test, y_train, y_test, metrics, task):
    if task == 2:
        clf = MultiOutputClassifier(LogisticRegression(max_iter=1000, multi_class='ovr'))
        param_grid = {'estimator__penalty': ['l2'],
                      'estimator__tol': [1e-3, 1e-4, 1e-5],
                      'estimator__solver': ['newton-cg', 'sag', 'saga', 'lbfgs']}
    else:
        clf = LogisticRegression(max_iter=1000, multi_class='ovr')
        param_grid = {'penalty': ['l2'],
                      'tol': [1e-3, 1e-4, 1e-5],
                      'solver': ['newton-cg', 'sag', 'saga', 'lbfgs']}
    # param_grid = {}
    cm, report, auc = apply_fold(X_train, X_test, y_train, y_test, clf, param_grid, metrics, task)
    # print(f'Best parameters: {clf.best_params_}')
    return cm, report, auc
