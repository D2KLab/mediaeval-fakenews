# FakeNews: Corona Virus and Conspiracies Multimedia Analysis Task
### Fighting against misinformation spreading

The FakeNews task explores various machine-learning approaches to automatically detect misinformation and its spreaders in social networks.

Spontaneous and intentional digital Fake News wildfires over online social media can be as dangerous as natural fires. A new generation of data mining and analysis algorithms is required for early detection and tracking of such information cascades. This task focuses on the analysis of tweets related to Coronavirus conspiracy theories in order to detect misinformation spreaders.


## Announcements
* **25 August 2021:** The development set is sent to the participants.


## Task Schedule
* 25 August: First development set release
* *TBA*: First intermediate test release
* *TBA*: Second development set release
* *TBA*: Second intermediate test release
* *TBA*: Third development set release
* 2 November: Final test set release
* 10 November: Runs due <!-- # Replace XX with your date. We suggest setting enough time in order to have enough time to assess and return the results by the Results returned deadline-->
* 15 November: Results returned  <!-- Replace XX with your date. Latest possible should be 15 November-->
* 22 November: Working notes paper due  <!-- Fixed. Please do not change. Exact date to be decided-->
* Beginning December: MediaEval 2021 Workshop  <!-- Fixed. Please do not change. Exact date to be decided-->



## Task Description
The FakeNews Detection Task offers three fake news detection subtasks on COVID-19-related conspiracy theories. The first subtask includes text-based fake news detection, the second subtask targets the detection of conspiracy theory topics, and the third subtask combines topic and conspiracy detection. All subtasks are related to misinformation disseminated in the context of the long-lasting COVID-19 crisis. We focus on conspiracy theories that assume some kind of nefarious actions by governments or other actors related to CODID-19, such as intentionally spreading the pandemic, lying about the nature of the pandemic, or using vaccines that have some hidden functionality and purpose.

***Text-Based Misinformation Detection***: In this subtask, the participants receive a dataset consisting of tweet text blocks in English related to COVID-19 and various conspiracy theories. **The participants are encouraged to build a multi-class classifier that can flag whether a tweet promotes/supports or discusses at least one (or many) of the conspiracy theories**. In the case if the particular tweet promotes/supports one conspiracy theory and just discusses another, the result of the detection for the particular tweet is experted to be equal to "**stronger**" class: promote/support in the given sample.

***Text-Based Conspiracy Theories Recognition***: In this subtask, the participants receive a dataset consisting of tweet text blocks in English related to COVID-19 and various conspiracy theories. **The main goal of this subtask is to build a detector that can detect whether a text in any form mentions or refers to any of the predefined conspiracy topics**.

***Text-Based Combined Misinformation and Conspiracies Detection***: In this subtask, the participants receive a dataset consisting of tweet text blocks in English related to COVID-19 and various conspiracy theories. **The goal of this subtask is to build a complex multi-labelling multi-class detector that for each topic from a list of predefined conspiracy topics can predict whether a tweet promotes/supports or just discusses that particular topic**.



#### Motivation and Background
Digital wildfires, i.e., fast-spreading inaccurate, counterfactual, or intentionally misleading information, can quickly permeate public consciousness and have severe real-world implications, and they are among the top global risks in the 21st century. While a sheer endless amount of misinformation exists on the internet, only a small fraction of it spreads far and affects people to a degree where they commit harmful and/or criminal acts in the real world. The COVID-19 pandemic has severely affected people worldwide, and consequently, it has dominated world news for months. Thus, it is no surprise that it has also been the topic of a massive amount of misinformation, which was most likely amplified by the fact that many details about the virus were unknown at the start of the pandemic. This task aims at the development of methods capable of detecting such misinformation. Since many different misinformation narratives exist, such methods must be capable of distinguishing between them. For that reason we consider a variety of well-known conspiracy theories related to COVID-19.  


#### Target Group
The task is of interest to researchers in the areas of online news, social media, multimedia analysis, multimedia information retrieval, natural language processing, and meaning understanding and situational awareness to participate in the challenge.


#### Data
The dataset contains several sets of tweet texts mentioning Corona Virus and different conspiracy theories. The dataset set consists of only English language posts and it contains a variety of long tweets with neutral, positive, negative, and sarcastic phrasing. The datasets is ***not balanced*** with respect to the number of samples of conspiracy-promoting and other tweets, and the number of tweets per each conspiracy class. The dataset items have been collected from Twitter during a period between 20th of January 2020 and 31st of July 2021, by searching for the Corona-virus-related keywords (e.g., "corona", "COVID-19", etc.) inside the tweets' text, followed by a search for keywords related to the conspiracy theories. Since not all tweets are available online, the partipants will be provided a full-text set of already downloaded tweets. In order to be compliant with the Twitter Developer Policy, only the members of the participants' participating temas are allowed to access and use the provided dataset. Distribution, publication, sharing and any form of usage of the provided data apart of the research purposes within the FakeNews task is strictly prohibited. A copy of the dataset in form of Tweet ID and annotations will be published after the end of MediaEval 2021.



#### Ground truth

The ground truth for the provided dataset was created by the team of well-motivated students and researchers using overlapping annotation process with the following cross-validation and verification by an independent assisting team.


#### Evaluation methodology

Evaluation will be performed using standard implementation of the multi-class generalization of the Matthews correlation coefficient (MCC, https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html) computed on the optimally threshold conspiracy promoting probabilities (threshold that yields the best MCC score).


## Task Steps

This year we will be running the task in a close-to-real-world problem solving mode. Instead of one-moment development dataset release, we split all the development dataset into three different. Consequentially, the task will be run in three-phase mode when participants encouraged to build their classifiers using the initially limited development set. After some development time the first test set will be send out and the participants are required to perform an intermediate evaluation of their algorithms submitting only one best run results. After the intermediate results submission, the second development set will be released allowing for further adjustment of the task solving algorithms.This is intermediate evaluation and additional development dataset release is repeated two times in total giving three development and two intermediate test sets. The final test set will be released shortly before the final runs submission deadline.

All-in-all, participation in this task involves the following steps:
1. Receive the first development data and design your approach using the first development set.
2. Receive the first test set, run your system on the first test data and submit one run results.
3. Receive the second development data and design your approach using the combination of the first and second development sets.
4. Receive the second test set, run your system on the second test data and submit one run results.
6. Receive the third development data and design your approach using the combination of the first, second and third development sets.
7. When the final test set is released, run your system on the final test data and submit up to 5 runs.
8. Receive your evaluation results (in terms of the official evaluation metric of the task), which you must report in the working notes paper.
9. Write and submit your working notes paper; different test runs of the final and intermediate test steps must all be described.


## Dataset

### Overview

The dataset consists of three development and three test sets containing full-text tweet bodies. Development sets are annotated differently for different subtasks: ***Text-Based Misinformation Detection*** subtask uses simple 3-class annotations, ***Text-Based Conspiracy Theories Recognition*** subtask uses multi-category binary annotations, and ***Text-Based Combined Misinformation and Conspiracies Detection*** task uses multi-category 3-class annotations.

### Ground Truth

In the ***Text-Based Misinformation Detection*** and ***Text-Based Combined Misinformation and Conspiracies Detection*** subtasks we use three different class labels to mark the tweet contents: *Promotes/Supports Conspiracy*, *Discusses Consparacy* and *Non-Conspiracy*.

* ***Promotes/Supports Conspiracy*** This class contains all tweets that promotes, supports, claim, insinuate some connection between COVID-19 and various conspiracies, such as, for example, the idea that 5G weakens the immune system and thus caused the current corona-virus pandemic; that there is no pandemic and the COVID-19 victims were actually harmed by radiation emitted by 5G network towers; ideas about an intentional release of the virus, forced or harmful vaccinations, or the virus being a hoax, etc. The crucial requirement is the claimed existence of some causal link.

* ***Discusses Consparacy*** This class contains all tweets that just mentioning the existing various conspiracies connected to COVID-19, or negating such a connection in clearly negative or sarcastic maneer.

* ***Non-Conspiracy*** This class contains all tweets not belonging to the previous two classes. Note that this also includes tweets that discuss COVID-19 pandemic itself.

In the ***Text-Based Conspiracy Theories Recognition*** and ***Text-Based Combined Misinformation and Conspiracies Detection*** subtasks we use nine different categories that corresponds to the most popular conspiracy theories: *Suppressed cures*, *Behaviour and Mind Control*, *Antivax*, *Fake virus*, *Intentional Pandemic*, *Harmful Radiation/ Influence*, *Population reduction*, *New World Order*, and *Satanism*.


### Data release

##### Development Sets

The following files are provided:

* `dev-<N>.zip` contains all the files of the N-th (first, second and third) development sets and is sent directly to the participants.

* `dev-<N>.zip/dev-<N>-task-1.csv` zipped file contains all the Tweets for ***Text-Based Misinformation Detection*** subtask
* `dev-<N>.zip/dev-<N>-task-2.csv` zipped file contains all the Tweets for ***Text-Based Conspiracy Theories Recognition*** subtask
* `dev-<N>.zip/dev-<N>-task-3.csv` zipped file contains all the Tweets for ***Text-Based Combined Misinformation and Conspiracies Detection*** subtask

The ***Text-Based Misinformation Detection*** subtask development dataset files are provided in CSV format with the following fields defined:
* *TweetID* - a FakeNews task internal tweet ID, do not match with the original tweet ID.
* *Class Label* - a class identifier value, 3 == ***Promotes/Supports Conspiracy***, 2 == ***Discusses Consparacy***, 1 == ***Non-Conspiracy***.
* *Tweet Text* - full tweet text block. Note that this field ends with the end of the CSV file line and it can contain extra commas that are not separators.


The ***Text-Based Conspiracy Theories Recognition*** subtask development dataset files are provided in CSV format with the following fields defined:
* *TweetID* - a FakeNews task internal tweet ID, do not match with the original tweet ID.
* *Binary Flag for Suppressed cures* - a flag indicating that the correcponding conspiracy theory is mentioned in the papticular tweet, 1 == mentioned, 0 == not mentioned (the same for the following Binary Flag fields). 
* *Binary Flag for Behaviour and Mind Control* - a flag indicating that the correcponding conspiracy theory is mentioned in the rapticular tweet.
* *Binary Flag for Antivax* - a flag indicating that the correcponding conspiracy theory is mentioned in the rapticular tweet.
* *Binary Flag for Fake virus** - a flag indicating that the correcponding conspiracy theory is mentioned in the rapticular tweet.
* *Binary Flag for Intentional Pandemic* - a flag indicating that the correcponding conspiracy theory is mentioned in the rapticular tweet.
* *Binary Flag for Harmful Radiation/ Influence* - a flag indicating that the correcponding conspiracy theory is mentioned in the rapticular tweet.
* *Binary Flag for Population reduction* - a flag indicating that the correcponding conspiracy theory is mentioned in the rapticular tweet.
* *Binary Flag for New World Order* - a flag indicating that the correcponding conspiracy theory is mentioned in the rapticular tweet.
* *Binary Flag for Satanism* - a flag indicating that the correcponding conspiracy theory is mentioned in the rapticular tweet.
* *Tweet Text* - full tweet text block. Note that this field ends with the end of the CSV file line and it can contain extra commas that are not separators.


The ***Text-Based Combined Misinformation and Conspiracies Detection*** subtask development dataset files are provided in CSV format with the following fields defined:
* *TweetID* - a FakeNews task internal tweet ID, do not match with the original tweet ID.
* *Class Label for Suppressed cures* - a class identifier value for the correcponding conspiracy theory in the papticular tweet, 3 == ***Promotes/Supports Conspiracy***, 2 == ***Discusses Consparacy***, 1 == ***Non-Conspiracy*** (the same for the following Class Label fields). 
* *Class Label for Behaviour and Mind Control* - a class identifier value for the correcponding conspiracy theory in the papticular tweet. 
* *Class Label for Antivax* - a class identifier value for the correcponding conspiracy theory in the papticular tweet. 
* *Class Label for Fake virus** - a class identifier value for the correcponding conspiracy theory in the papticular tweet. 
* *Class Label for Intentional Pandemic* - a class identifier value for the correcponding conspiracy theory in the papticular tweet. 
* *Class Label for Harmful Radiation/ Influence* - a class identifier value for the correcponding conspiracy theory in the papticular tweet. 
* *Class Label for Population reduction* - a class identifier value for the correcponding conspiracy theory in the papticular tweet. 
* *Class Label for New World Order* - aa class identifier value for the correcponding conspiracy theory in the papticular tweet. 
* *Class Label for Satanism* - a class identifier value for the correcponding conspiracy theory in the papticular tweet. 
* *Tweet Text* - full tweet text block. Note that this field ends with the end of the CSV file line and it can contain extra commas that are not separators.


*All CSV files use comma as a field separator. Note that fields that corresponds to *Tweet Text* ends with the end of the CSV file line and can contain extra commas that are not separators, but parts of the tweet text content*.

*All CSV files are UTF-8 encoded and stored in Linux-style text file format using only one line ending character (0x0A in hex, '\n' in C/C++)*.


#### Test Sets

**TBA**


## Submission

### Run Submissions

**TBA**


#### References and recommended reading

***General***

[1] Nyhan, Brendan, and Jason Reifler. 2015. [Displacing misinformation about events: An experimental test of causal corrections](https://www.cambridge.org/core/journals/journal-of-experimental-political-science/article/displacing-misinformation-about-events-an-experimental-test-of-causal-corrections/69550AB61F4E3F7C2CD03532FC740D05#). Journal of Experimental Political Science 2, no. 1, 81-93.

***Twitter data collection and analysis***

[2] Burchard, Luk, Daniel Thilo Schroeder, Konstantin Pogorelov, Soeren Becker, Emily Dietrich, Petra Filkukova, and Johannes Langguth. 2020. [A Scalable System for Bundling Online Social Network Mining Research](https://ieeexplore.ieee.org/document/9336577). In 2020 Seventh International Conference on Social Networks Analysis, Management and Security (SNAMS), IEEE, 1-6.

[3] Schroeder, Daniel Thilo, Konstantin Pogorelov, and Johannes Langguth. 2019. [FACT: a Framework for Analysis and Capture of Twitter Graphs](https://ieeexplore.ieee.org/document/8931870). In 2019 Sixth International Conference on Social Networks Analysis, Management and Security (SNAMS), IEEE, 134-141.

[4] Achrekar, Harshavardhan, Avinash Gandhe, Ross Lazarus, Ssu-Hsin Yu, and Benyuan Liu. 2011. [Predicting flu trends using twitter data](https://ieeexplore.ieee.org/document/5928903). In 2011 IEEE conference on computer communications workshops (INFOCOM WKSHPS), IEEE, 702-707.

[5] Chen, Emily, Kristina Lerman, and Emilio Ferrara. 2020. [Covid-19: The first public coronavirus twitter dataset](https://arxiv.org/abs/2003.07372v1?utm_source=feedburner&utm_medium=feed&utm_campaign=Feed%3A+CoronavirusArXiv+%28Coronavirus+Research+at+ArXiv%29). arXiv preprint arXiv:2003.07372.

[6] Kouzy, Ramez, Joseph Abi Jaoude, Afif Kraitem, Molly B. El Alam, Basil Karam, Elio Adib, Jabra Zarka, Cindy Traboulsi, Elie W. Akl, and Khalil Baddour. 2020. [Coronavirus goes viral: quantifying the COVID-19 misinformation epidemic on Twitter](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7152572/). Cureus 12, no. 3.

***Natural language processing***

[7] Bourgonje, Peter, Julian Moreno Schneider, and Georg Rehm. 2017. [From clickbait to fake news detection: an approach based on detecting the stance of headlines to articles](https://www.aclweb.org/anthology/W17-4215/). In Proceedings of the 2017 EMNLP Workshop: Natural Language Processing meets Journalism, 84-89.

[8] Imran, Muhammad, Prasenjit Mitra, and Carlos Castillo. 2016. [Twitter as a lifeline: Human-annotated twitter corpora for NLP of crisis-related messages](https://arxiv.org/abs/1605.05894). arXiv preprint arXiv:1605.05894.

***Information spreading***

[9] Liu, Chuang, Xiu-Xiu Zhan, Zi-Ke Zhang, Gui-Quan Sun, and Pak Ming Hui. 2015. [How events determine spreading patterns: information transmission via internal and external influences on social networks](https://iopscience.iop.org/article/10.1088/1367-2630/17/11/113045/pdf). New Journal of Physics 17, no. 11.

***Online news sources analysis***

[10] Pogorelov, Konstantin, Daniel Thilo Schroeder, Petra Filkukova, and Johannes Langguth. 2020. [A System for High Performance Mining on GDELT Data](https://ieeexplore.ieee.org/document/9150419). In 2020 IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW), IEEE, 1101-1111.


#### Task Organizers
* Konstantin Pogorelov, Simula Research laboratory (Simula), Norway, konstantin (at) simula.no
* Johannes Langguth, Simula Research laboratory (Simula), Norway, langguth (at) simula.no
* Daniel Thilo Schroeder, Simula Research laboratory (Simula), Norway


## Acknowledgements

This work was funded by the Norwegian Research Council under contracts #272019 and #303404 and has benefited from the Experimental Infrastructure for Exploration of Exascale Computing (eX3), which is financially supported by the Research Council of Norway under contract #270053. We also acknowledge support from Michael Kreil in the collection of Twitter data.
