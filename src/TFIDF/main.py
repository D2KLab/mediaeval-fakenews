import argparse
import math
from methods import *
from utils import *


def create_matrices(ids, texts, args):
    if args.task == 1:
        #tfidf_matrix = compute_tfidf(texts, ids, args.hash_size, args.ngrams)
        #sp.save_npz('tfidf_matrices/scratch_matrix.npz', tfidf_matrix)
        tfidf_matrix_vect = compute_tfidf_vect(texts, args.ngrams)
        sp.save_npz('tfidf_matrices/vect_matrix.npz', tfidf_matrix_vect)
    else:
        #tfidf_matrix = compute_tfidf(texts, ids, args.hash_size, args.ngrams)
        #sp.save_npz('tfidf_matrices/scratch_matrix_' + str(args.task) + '.npz', tfidf_matrix)
        tfidf_matrix_vect = compute_tfidf_vect(texts, args.ngrams)
        sp.save_npz('tfidf_matrices/vect_matrix_' + str(args.task) + '.npz', tfidf_matrix_vect)


def create_test_matrices(ids, texts, test, args):
    X_train, X_test = compute_test_tfidf(texts, test, args.ngrams)
    sp.save_npz('tfidf_matrices/train_matrix.npz', X_train)
    sp.save_npz('tfidf_matrices/test_matrix.npz', X_test)


# Retrieve the previously stored matrices
def read_matrix(task):
    if task == 1:
        matrix = sp.load_npz("tfidf_matrices/vect_matrix.npz")
    else:
        matrix = sp.load_npz("tfidf_matrices/vect_matrix_" + str(args.task) + ".npz")
    return matrix.todense()


def define_algorithms():
    algs = [GaussianNB_Classifier, DecisionTree_Classifier, BernoulliNB_Classifier, RandomForest_Classifier,
            SVC_Classifier, SGD_Classifier, AdaBoost_Classifier, Ridge_Classifier,
            LogisticRegression_Classifier]
    names = ['Gaussian NB', 'Decision Tree', 'Bernoulli NB', 'Random forest', 'SVM', 'SGD', 'AdaBoost',
             'Ridge', 'Logistic Regression']
    ret = {}
    for i, n in enumerate(names):
        ret[n] = algs[i]
    return ret


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=int, default=2, help="Perform the algorithms on task 1 or 2")
    #parser.add_argument('--hash-size', type=int, default=int(math.pow(2, 25)), help="hash size for TFIDF matrix generated from scratch")
    parser.add_argument('--ngrams', type=int, default=2, help="N for ngrams")
    parser.add_argument('--command', type=int, default=1, help="0: create matrices - 1: run algorithms")
    # parser.add_argument('--model', type=str, default=None)
    args = parser.parse_args()
    filename = '../data_clean/dev-full-task-' + str(args.task) + '-clean.csv'
    #ids, labels, texts = read_tweets(filename, args.task)
    df = pd.read_csv(filename)
    texts = []
    for index, row in df.iterrows():
        texts.append(row['tweet'])
    texts = np.array(texts)
    ids = df['ids'].to_numpy()
    if args.task == 1:
        labels = df['1'].to_numpy()
    else:
        labels = df[['1', '2', '3', '4', '5', '6', '7', '8', '9']].to_numpy()
    ids, test_texts = read_tweets_text('../test/test-task-' + str(args.task) + '.csv')
    if args.command == 0:
        create_test_matrices(ids, texts, test_texts, args)
    elif args.command == 1:
        tfidf_matrix = read_matrix(args.task)
        filename = "./task" + str(args.task) + "/dev-full-split-4.csv"
        indices = pd.read_csv(filename)['ids'].to_numpy()
        indices -= 1
        # plot_data(tfidf_matrix, labels)
        metrics = define_scores()
        algs = define_algorithms()
        for alg in algs.keys():
            print("*"*25 + " " + alg + " " + "*"*25)
            results, roc, mcc = method_applier(tfidf_matrix, labels, algs[alg], metrics, args.task, indices)
            print(f"ACC: {results}")
            print(f"AUC: {roc}")
            print(f"MCC: {mcc}")
