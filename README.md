# FakeNews: Corona Virus and Conspiracies Multimedia Analysis Task

This repository contains D2KLab participation to the MediaEval FakeNews challenge 2021 [[1]](#1).

The code implementation of the three approaches are available in [./src/](./src/).

An inference notebook is available in [./inference/inference.ipynb](./inference/inference.ipynb).

All models are availailable for download [here](https://mediaeval-fakenews.tools.eurecom.fr/) and should be put in a directory ./models to run the inference notebook.

## Paper
You can read our paper [here](https://2021.multimediaeval.com/paper65.pdf)

## Citation
```
Youri Peskine, Giulio Alfarano, Ismail Harrando, Paolo Papotti, Raphaël Troncy.
Detecting COVID-19-Related Conspiracy Theories in Tweets.
MediaEval 2021 - MediaEval Multimedia Evaluation benchmark. Workshop, Dec 2021, Online
```

## Approach

In order to tackle this challenge, we studied three different kind of approaches. The first uses a combination of TFIDF and machine learning algorithms. The second approach uses Natural Language Inference (NLI) combined with metadata from Wikipedia. The third approach aims at fine-tuning transformer-based models.
This last approach was the most performing one and got the best results on all the tasks amongst all the participants.


Our runs are available in [./runs/](./runs/).
 - Run 1 is TFIDF
 - Run 2 is CTBert
 - No run 3
 - Run 4 is task-3-CTBert
 - Run 5 is late fusion ensembling


## References
<a id="1">[1]</a> 
K. Pogorelov, D. Thilo Schroeder, S. Brenner, and J. Langguth (2021), FakeNews: Corona Virus and Conspiracies Multimedia Analysis Task at MediaEval
2021Codes
