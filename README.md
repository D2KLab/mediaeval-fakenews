# FakeNews: Corona Virus and Conspiracies Multimedia Analysis Task

This repository contains the D2KLab participation to the [MediaEval 2021 FakeNews Task](https://multimediaeval.github.io/editions/2021/tasks/fakenews/) and [MediaEval 2022 FakeNews Task](https://multimediaeval.github.io/editions/2022/tasks/fakenews/) .

# 2022
This year's ''FakeNews Detection'' task aims at detecting 9 named conspiracy theories in tweets, as well as classifying misinformation spreaders in a user interaction graph.

We propose a Transformer-based approach (CT-BERT) to tackle task 1, and node embedding (node2vec) to tackle task 2. We then concatenate both text and graph features and perform classification for task 3. The code implementation is available in [./2022/src/](./2022/src/).

## Approach

In order to tackle this challenge, we studied text-classification transformer-models for task 1 and 3, and node-classification models for task 2 and 3. Our approach leverages multiple **CT-BERT** models for text-classification and **node2vec** in combination with simple classifiers (MLP, RF) for node-classification. In all experiments, we split the data into 5 stratified cross-validation sets.

## Results (2022)
The results obtained with our approach are summarised in the following figure:


## Main Takaways
 - Even though we had more data this year, we did not see an increase in performance
 - Some conspiracies (*Harmful Radiation/Influence* or *New World Order*) are easier to detect than others (*Antivax*). In this example, *Antivax* has four times more data than *Harmful Radiation/Influence* but performs significantly worse.
 - Graph-related tasks are challenging and there is room for improvement.
 - Other approach could have been studied (e.g. GNNs)



# 2021
We proposed three approaches for which the code implementation are available in [./2021/src/](./2021/src/) for the ones who would like to retrain our models.

An inference notebook is also directly available in [./2021/inference/inference.ipynb](./2021/inference/inference.ipynb). All models are available for download at https://mediaeval-fakenews.tools.eurecom.fr/index.html

The path to the models needs to be specified in the **Input** cell of the inference notebook.

## Citation
```
Youri Peskine, Giulio Alfarano, Ismail Harrando, Paolo Papotti, Raphaël Troncy.
Detecting COVID-19-Related Conspiracy Theories in Tweets.
In Multimedia Benchmark Workshop (MediaEval 2021), 13-15 December 2021, Online.
https://2021.multimediaeval.com/paper65.pdf
```

## Approach

In order to tackle this challenge, we studied three different kind of approaches. The first uses a combination of TFIDF and machine learning algorithms. The second approach uses Natural Language Inference (NLI) combined with metadata from Wikipedia. The third approach aims at fine-tuning transformer-based models.
This last approach was the most performing one and got the best results on all the tasks amongst all the participants.



## Results (2021)
The results for our 3 approaches on a validation set and on the test set are summarized on this figure.
Our 2021 runs are available in [./2021/runs/](./2021/runs/).
 - Run 1 is TFIDF
 - Run 2 is CTBert
 - No run 3
 - Run 4 is task-3-CTBert
 - Run 5 is late fusion ensembling

![plot](./results.png)

## Requirements
```
python==3.8
torch==1.6.0
transformers==3.1.0
pandas==1.3.3
numpy==1.22.3
emoji==0.5.3
notebook
scikit-learn
```
