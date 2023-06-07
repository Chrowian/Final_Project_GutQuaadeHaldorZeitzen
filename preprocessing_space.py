import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.data.path.append("<path_to_nltk_data>")

stop_words = set(stopwords.words('english'))

# Define the special characters you want to add spaces before and after
special_chars = r'([“!@#$%^&*()_\-+=<>?.,:;\"/\\])'

def preprocess_space(dataframe):
    """
    :param - dataframe: pandas dataframe X (text)
    :return: pandas dataframe X (text) WITH SPACES around special characters
    """
    dataframe = dataframe.str.replace(special_chars, r' \1 ', regex=True)
    return dataframe


def remove_stop_words(text):
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in text.split() if word.lower() not in stop_words]
    return ' '.join(filtered_tokens)

def remove_stopwords(dataframe):
    """
    :param - dataframe: pandas dataframe X (text)
    :return: pandas dataframe X (text) WITHOUT STOPWORDS
    """
    nltk.data.path.append("<path_to_nltk_data>")
    df = dataframe.apply(remove_stop_words)
    return df