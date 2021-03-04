import os
import pickle as pk
from urllib.request import urlretrieve

import spacy
import swifter
import pandas as pd
import pycaret.classification as pycc

nlp = spacy.load('en_core_web_lg')

def download_liu_training_data():
    """
    Checks if Liu's training data exists; if not, downloads it.
    Source: https://github.com/tlkh/text-emotion-classification
    """
    if not os.path.exists('data/text_emotion.csv'):
        os.mkdir('data')
        url = 'https://raw.githubusercontent.com/tlkh/text-emotion-classification/master/dataset/data/text_emotion.csv'
        urlretrieve(url, 'data/text_emotion.csv')


def get_saif_files():
    """
    Gets filenames for Saif's dataset.
    """
    emotions = ['anger', 'fear', 'joy', 'sadness']
    files = [f'data/{e}.tsv' for e in emotions]
    return files


def download_saif_training_data():
    """
    Checks if Saif's training data exists; if not, downloads it.
    Source: http://saifmohammad.com/WebPages/EmotionIntensity-SharedTask.html
    """
    files = get_saif_files()
    urls = [f'http://saifmohammad.com/WebDocs/EmoInt%20Train%20Data/{e}-ratings-0to1.train.txt' for e in emotions]
    for u, f in zip(urls, files):
        path = os.path.join('data', f)
        if not os.path.exists(path):
            urlretrieve(u, f)


def load_prepare_liu_data():
    """
    Loads and prepares Liu's data. Combines similar categories, and
    maps labels to the same as Saif's data (anger, fear, joy, sadness).
    """
    df = pd.read_csv('data/text_emotion.csv')
    df['sentiment'].replace({'boredom': 'neutral',
                            'hate': 'anger',
                            'fun': 'happiness',
                            'empty': 'neutral',
                            'enthusiasm': 'happiness',
                            'love': 'happiness'}, inplace=True)
    df['sentiment'].replace({'happiness': 'joy',
                            'worry': 'fear',
                            'surprise': 'joy',
                            'relief': 'joy'},
                            inplace=True)

    df.drop(['tweet_id', 'author'], axis=1, inplace=True)
    df.rename(columns={'sentiment': 'emotion', 'content': 'tweet'}, inplace=True)
    df = df[['tweet', 'emotion']]
    return df


def load_prepare_saif_data(threshold=0.25):
    """
    Loads and prepares saif's data.

    Parameters
    ----------
    threshold : float
        Only data with intensities equal to or 
        above this threshold will be kept (range 0-1).

    Returns
    -------
    DataFrame : pd.DataFrame
        Concatenated tweets with labels as a pandas DataFrame.
    """
    files = get_saif_files()
    df = pd.concat([pd.read_csv(f, sep='\t', index_col=0, names=['tweet', 'emotion', 'intensity']) for f in files], axis=0)
    df = df[df['intensity'] >= threshold]
    df.drop('intensity', axis=1, inplace=True)
    return df


def load_combine_saif_liu_data():
    """
    Loads and combines dataframes for Liu's and Saif's data.
    """
    liu_df = load_prepare_liu_data()
    saif_df = load_prepare_saif_data()
    return pd.concat([liu_df, saif_df], axis=0).reset_index(drop=True)


def get_text_vector(x):
    """
    Gets text vector from input string.
    """
    t = nlp(x)
    return t.vector


if __name__ == "__main__":
    df = load_combine_saif_liu_data()
    sample=False
    if sample:
        df = df.groupby('emotion',
                group_keys=False).apply(lambda x: x.sample(min(len(x), 100)))


    vector_df_filepath = 'data/vector_df.csv'
    if os.path.exists(vector_df_filepath):
        vector_df = pd.read_csv(vector_df_filepath)
    else:
        vectors = df['tweet'].swifter.apply(get_text_vector)
        vector_df = pd.DataFrame(vectors.array,
                        columns=[f'v{r}' for r in range(vectors.iloc[0].shape[0])])
        vector_df.to_csv(vector_df_filepath)

    data_df = pd.concat([vector_df, pd.DataFrame(df['emotion']).reset_index(drop=True)],
                    axis=1)

    models = pycc.setup(data=data_df, target='emotion', session_id=123)

    best = pycc.compare_models()

    pycc.save_model(best, 'lr_emotion.model')
    with open('pycc_setup.pk', 'wb') as f:
        pk.dump(models, f)