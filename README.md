# emotion_classification
Trying different ML algorithms and text embeddings for emotion classification.

Uses word vectors to train a classifier for emotion prediction from text. Compares ML algos:

- XGBoost
- lightGBM
- catboost
- neural network
- ensemble of all

Compares word vectors:
- spacy
- GloVe
- 

# Quickstart
Install virtualenv:

`pip install -r requirements.txt`

Install word vectors:

`python -m spacy download en_core_web_lg`

To classify the emotion of text, run:

``

data: https://raw.githubusercontent.com/tlkh/text-emotion-classification/master/dataset/data/text_emotion.csv
from:
https://github.com/tlkh/text-emotion-classification

other potential datasets:
https://github.com/lukasgarbas/nlp-text-emotion/tree/master/data
http://saifmohammad.com/WebPages/EmotionIntensity-SharedTask.html
