<img src="compass_rose.jpeg" />

This is a prototyping repo of text-mining and search methods developed for the COVID-19 document dataset from the Allen Institute for Ai.
Evaluation/tasks are still forthcoming from the Kaggle, but most of the problems seem more like SWE than ML:
* what use cases address the needs of COVID-19 researchers
* what search methods or frameworks can retrieve/extract information as effectively as possible


This repo is a demonstration of vector-space language modeling with [Gensim](https://radimrehurek.com/gensim/), to support the following use cases:
* semantic keyword search:
  * ranking relevant research papers using document vectors
  * 'also see...' suggestions
  * user-parameterized search: letting researchers find documents based on their own criteria, e.g. not the search-author's criteria.
* knowledge/expert mapping:
  * network modeling of author expertise using language modeling. Essentially, given author X, link other author's whose expertise is most similar.
  * navigation/search of the expert-graph
  * knowledge-cluster identification: e.g., who should you give your advertising fortune to
* cross-language search and paper translation


Most of these are easily implemented/supported by vector-space neural modeling methods like Gensim.
I do not have the resources to work on this project diligently, so feel free to use it for reference.
If you would like development work, I'll be glad to help in any way I can.

Links:
* https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/tasks
* https://www.whitehouse.gov/briefings-statements/call-action-tech-community-new-machine-readable-covid-19-dataset/
* https://pages.semanticscholar.org/coronavirus-research


To use the repo in its current state:
1) Download the COVID-19 commercial use dataset to the dataset/ folder: dataset/comm_use_dataset/*
2) Cd into src/ and run: python3 covid.py


This will build a gensim Word2Vec language model, and save it to the models/ folder.
Once built, the entire dataset is read into memory (got RAM?), each paper is converted to a document vector by averaging
its terms. A command line query loop then accepts input terms from the user, and returns the most similar terms/documents 
per the query using cosine-similarity.


COVID-19 is *not the end of the world*, and for that matter nothing is, when we inspire confidence in others, think positively, and seek ways to contribute. [So let's get to work.](https://www.youtube.com/watch?v=cCYGyg1H56s)


