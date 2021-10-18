import argparse
import numpy as np
import torch
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, matthews_corrcoef, roc_auc_score
from transformers import BertForSequenceClassification, BertTokenizer, RobertaTokenizer, RobertaForSequenceClassification, TrainingArguments, Trainer


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
                labels.append(int(row[1])-1)
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
                #labels.append([-1 if int(el) == 0 else 1 for el in row[1:10]])
                labels.append([int(el)-1 for el in row[1:10]])
                tweets.append(text)
    return ids, labels, tweets


def one_hot_encoding(labels):
    dictionary = {0: [0, 0, 1],
                  1: [0, 1, 0],
                  2: [1, 0, 0]}
    enc_labels = []
    for el in labels:
        enc_labels.append(dictionary[el])
    return np.array(enc_labels)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds)
    acc = accuracy_score(labels, preds)
    roc = roc_auc_score(one_hot_encoding(labels), one_hot_encoding(preds), average='weighted', multi_class='ovr')
    mcc = matthews_corrcoef(labels, preds)
    class_rep = classification_report(labels, preds, target_names=['Non-Conspiracy', 'Discusses Conspiracy',
                                                                   'Promotes/Supports Conspiracy'], output_dict=True)
    print(class_rep)
    print(f"ACC:{acc}\n RECALL:{recall}\n PREC:{precision}\n F1:{f1}")
    print(f"ROC:{roc}\n MCC:{mcc}")
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'roc': roc,
        'mcc': mcc,
        'class_rep': class_rep
    }


def model_trainer(model_path, n_labels, train_set=None, test_set=None):
    # Load BertForSequenceClassification, the pretrained BERT model with a single linear classification layer on top.
    if train_set is not None:
        #model = BertForSequenceClassification.from_pretrained("bert-large-uncased", num_labels=n_labels, return_dict=True)
        model = RobertaForSequenceClassification.from_pretrained("roberta-large", num_labels=n_labels, return_dict=True)
    else:
        #model = BertForSequenceClassification.from_pretrained(model_path, num_labels=n_labels, return_dict=True)
        model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=n_labels, return_dict=True)
    model.cuda()

    training_args = TrainingArguments(
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=16,   # batch size for evaluation
        # warmup_steps=0,                # number of warmup steps for learning rate scheduler
        logging_dir='./output_model/logs',
        output_dir=model_path,
        #save_strategy='epoch'
    )

    if train_set is not None:
        trainer = Trainer(
            model=model,                         # the instantiated 🤗 Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=train_set,          # training dataset
            eval_dataset=test_set,
            compute_metrics=compute_metrics
        )
    else:
        trainer = Trainer(
            model=model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            eval_dataset=test_set,
            compute_metrics=compute_metrics
        )
    return trainer


class BERTdataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, use_label=True):
        self.encodings = encodings
        self.labels = labels
        self.use_label = use_label

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.use_label:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def main_train(args, X_train, y_train, X_val, y_val):
    #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large', do_lower_case=True)
    text_train = tokenizer(X_train, padding=True, truncation=True)
    train_dataset = BERTdataset(text_train, y_train)
    text_val = tokenizer(X_val, padding=True, truncation=True)
    val_dataset = BERTdataset(text_val, y_val)
    text_test = tokenizer(X_test, padding=True, truncation=True)
    test_dataset = BERTdataset(text_test, y_test)
    model_path = './output_model/'
    trainer = model_trainer(model_path=model_path, n_labels=len(set(labels)), train_set=train_dataset, test_set=val_dataset)
    trainer.train()
    trainer.save_model()
    scores = trainer.evaluate()
    print(scores['eval_roc'])
    print(scores['eval_mcc'])
    print(scores['eval_class_rep'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=int, default=1)
    args, _ = parser.parse_known_args()

    # tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
    filename = 'dev-1/dev-1-task-' + str(args.task) + '.csv'
    ids, labels, lines = read_tweets(filename, args.task)
    if args.task == 1:
        X_train, X_test, y_train, y_test = train_test_split(lines, labels, stratify=labels, test_size=0.25,
                                                            random_state=42)
        #X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=0.25,
        #                                                  random_state=0)
    else:
        X_train, X_test, y_train, y_test = train_test_split(lines, labels, test_size=0.25, random_state=0)
        # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=0)

    main_train(args, X_train, y_train, X_test, y_test)

