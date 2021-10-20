import argparse
import math
from methods import *
from utils import *


# Generate and store the TFIDF matrices both from scratch and through the Vectorizer function
# The matrices are ready in tdidf_matrices folder
def create_matrices(ids, texts, args):
    if args.task == 1:
        tfidf_matrix = compute_tfidf(texts, ids, args.hash_size, args.ngrams)
        sp.save_npz('tfidf_matrices/scratch_matrix.npz', tfidf_matrix)
        tfidf_matrix_vect = compute_tfidf_vect(texts, args.ngrams)
        sp.save_npz('tfidf_matrices/vect_matrix.npz', tfidf_matrix_vect)
    else:
        tfidf_matrix = compute_tfidf(texts, ids, args.hash_size, args.ngrams)
        sp.save_npz('tfidf_matrices/scratch_matrix_' + str(args.task) + '.npz', tfidf_matrix)
        tfidf_matrix_vect = compute_tfidf_vect(texts, args.ngrams)
        sp.save_npz('tfidf_matrices/vect_matrix_' + str(args.task) + '.npz', tfidf_matrix_vect)


# Retrieve the previously stored matrices
def read_matrix(m, task):
    if task == 1:
        if m == 1:
            matrix = sp.load_npz("tfidf_matrices/vect_matrix.npz")
        elif m == 0:
            matrix = sp.load_npz("tfidf_matrices/scratch_matrix.npz")
        else:
            exit(1)
    else:
        if m == 1:
            matrix = sp.load_npz("tfidf_matrices/vect_matrix_" + str(args.task) + ".npz")
        elif m == 0:
            matrix = sp.load_npz("tfidf_matrices/scratch_matrix_" + str(args.task) + ".npz")
        else:
            exit(1)
    return matrix


def define_algorithms():
    algs = [DecisionTree_Classifier, GaussianNB_Classifier, BernoulliNB_Classifier, RandomForest_Classifier,
            SVC_Classifier, SGD_Classifier, AdaBoost_Classifier, Ridge_Classifier,
            LogisticRegression_Classifier]
    names = ['Decision Tree', 'Gaussian NB', 'Bernoulli NB', 'Random forest', 'SVM', 'SGD', 'AdaBoost',
             'Ridge', 'Logistic Regression']
    ret = {}
    for i, n in enumerate(names):
        ret[n] = algs[i]
    return ret


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=int, default=2, help="Perform the algorithms on task 1 or 2")
    parser.add_argument('--hash-size', type=int, default=int(math.pow(2, 24)), help="hash size for TFIDF matrix generated from scratch")
    parser.add_argument('--ngrams', type=int, default=2, help="N for ngrams")
    parser.add_argument('--matrix', type=int, default=1, help="0: from scratch - 1: through vectorizer")
    parser.add_argument('--command', type=int, default=1, help="0: create matrices - 1: run algorithms")
    # parser.add_argument('--model', type=str, default=None)
    args = parser.parse_args()
    filename = '../dev-1/dev-1-task-' + str(args.task) + '.csv'
    ids, labels, texts = read_tweets(filename, args.task)
    if args.command == 0:
        create_matrices(ids, texts, args)
    elif args.command == 1:
        tfidf_matrix = read_matrix(args.matrix, args.task)
        # plot_data(tfidf_matrix, labels)
        metrics = define_scores()
        algs = define_algorithms()
        for alg in algs.keys():
            print("*"*25 + " " + alg + " " + "*"*25)
            results, roc, mcc = method_applier(tfidf_matrix, labels, algs[alg], metrics, args.task)
            print(results)
            print(f"AUC: {roc}")
            print(f"MCC: {mcc}")