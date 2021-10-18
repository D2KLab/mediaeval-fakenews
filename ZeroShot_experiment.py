from transformers import pipeline
import numpy as np
import csv
import argparse
import itertools
from sklearn.metrics import roc_auc_score, matthews_corrcoef
from tqdm import tqdm


def read_tweets(filename, task):
    ids = []
    labels = []
    tweets = []
    with open(filename, 'r', encoding="utf8") as f:
        if task == 1:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                if len(row) > 3:
                    text = ''
                    for i in range(2, len(row)):
                        if i != 2:
                            text += ', ' + row[i]
                        else:
                            text += row[i]
                else:
                    text = row[2]
                ids.append(int(row[0])-1)
                labels.append(int(row[1]))
                tweets.append(text)
        elif task == 2:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                if len(row) > 11:
                    text = ''
                    for i in range(10, len(row)):
                        if i != 10:
                            text += ', ' + row[i]
                        else:
                            text += row[i]
                else:
                    text = row[10]
                ids.append(int(row[0]) - 1)
                labels.append([int(el) for el in row[1:10]])
                tweets.append(text)
    return ids, labels, tweets


# MCC is computed for each sample's truth-predicted pair and then the mean is computed
def computeMCC(y_test, y_pred):
    value = 0
    for y1, y2 in zip(y_test, y_pred):
        value += matthews_corrcoef(y1, y2)
    mcc = value / len(y_test)
    return mcc


# MCC score is computed for each label
def computeMCCclass(y_test, y_pred):
    mccs = []
    for i in range(len(y_test[0,:])):
        mccs.append(matthews_corrcoef(y_test[:,i], y_pred[:,i]))
    return mccs


# This function defines the vector of predicted labels based on the scores from the ZS classifier and the user-defined threshold
def obtain_pred(par, scores, order_labels, lab_dict):
    pred_label = []
    for score in scores:
        if score >= par:
            pred_label.append(lab_dict[order_labels[0]])
        else:
            break
    if len(pred_label) != 0:
        pred = np.array(pred_label)
        y_pred = []
        for i in range(len(pred[0, :])):
            y_pred.append(sum(pred[:, i]))
        # y_pred = np.array([int(x + y) for x, y in zip(pred_label[0], pred_label[1])])
        # y_pred[y_pred > 1] = 1
        # y_pred.append(0)
    else:
        y_pred = [0] * 9
        # y_pred.append(1)
    return y_pred


def main(filename, task, par):
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    candidate_labels = ['suppressed cures', 'mind control', 'antivax', 'fake virus', 'intentional pandemic',
                        'harmful radiation/influence', 'population reduction', 'new world order', 'satanism']
    lab_dict = {}
    for i, lab in enumerate(candidate_labels):
        cod = [0]*9
        cod[i] = 1
        lab_dict[lab] = cod.copy()
    ids, labels, texts = read_tweets(filename, task)
    y_true = []
    y_pred = []
    for lab, tweet in tqdm(zip(labels, texts)):
        res = classifier(tweet, candidate_labels, multi_label=True)
        order_labels = res['labels']
        scores = res['scores']
        y_pred.append(obtain_pred(par, scores, order_labels, lab_dict))
        ''' if sum(lab) == 0:
            lab.append(1)
        else:
            lab.append(0)'''
        y_true.append(lab)
    try:
        print(f"ROC: {roc_auc_score(y_true, y_pred, average='weighted', multi_class='ovr')}")
    except ValueError:
        print(y_true)
        print(y_pred)
    print(f"MCC: {computeMCCclass(np.array(y_true), np.array(y_pred))}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=int, default=2, help="In this case it is useful only to retrieve the right data through the read_tweets function")
    parser.add_argument('--threshold', type=float, default=0.6, help="User-defined threshold")
    args = parser.parse_args()
    filename = 'dev-1/dev-1-task-' + str(args.task) + '.csv'
    main(filename, args.task, args.threshold)
