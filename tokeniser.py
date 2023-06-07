# Script for tokenising the titles and text of the articles, once they have been cleaned.

from typing import List
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import pandas as pd
from transformers import BertTokenizer
import tensorflow as tf

from keras.preprocessing.text import Tokenizer
import pandas as pd

def tokenize_vocab(df: pd.DataFrame, 
                       title_column: str = 'title', 
                       text_column: str = 'text',
                       custom_vocab: dict = None,
                       max_length: int = 100) -> pd.DataFrame:
    """
    Tokenizes and pads the 'title' and 'text' columns of a DataFrame using a custom vocabulary.

    Parameters:
    df (pd.DataFrame): DataFrame containing the text to tokenize and pad.
    title_column (str): Name of the column containing titles. Default is 'title'.
    text_column (str): Name of the column containing texts. Default is 'text'.
    custom_vocab (dict): A dictionary defining a custom vocabulary, with words as keys and tokens as values.
                         If None, the vocabulary will be learned from the texts. Default is None.
    max_length (int): Maximum length for all sequences. If a sequence is shorter than the max length, 
                      it will be padded, if it is longer, it will be truncated. Default is 100.

    Returns:
    pd.DataFrame: A DataFrame with the tokenized and padded 'title' and 'text' columns.
    """
    # Initialize tokenizer
    tokenizer = Tokenizer(filters='')
    
    if custom_vocab is not None:
        # Set word_index property
        tokenizer.word_index = custom_vocab
        tokenizer.num_words = len(custom_vocab) + 1
    
    else:
        # Learn the vocabulary from the texts
        titles = df[title_column]
        texts = df[text_column]
        tokenizer.fit_on_texts(pd.concat([titles, texts]))

    # Convert texts to sequences
    sequences_titles = tokenizer.texts_to_sequences(df[title_column])
    sequences_texts = tokenizer.texts_to_sequences(df[text_column])

    # Pad sequences
    padded_titles = pad_sequences(sequences_titles, maxlen=max_length, padding='post')
    padded_texts = pad_sequences(sequences_texts, maxlen=max_length, padding='post')

    # Create new DataFrame with tokenized and padded sequences
    df_tokenized = df.copy()
    df_tokenized[title_column] = list(padded_titles)
    df_tokenized[text_column] = list(padded_texts)

    return df_tokenized

def tokenize_dataframe(df: pd.DataFrame, 
                       title_column: str = 'title', 
                       text_column: str = 'text',
                       vocab_size: int = 50000,
                       max_length: int = 100) -> pd.DataFrame:
    """
    Tokenizes and pads the 'title' and 'text' columns of a DataFrame.

    Parameters
    ----------
    df (pd.DataFrame): DataFrame containing the text to tokenize and pad.
    title_column (str): Name of the column containing titles. Default is 'title'.
    text_column (str): Name of the column containing texts. Default is 'text'.
    vocab_size (int): Maximum number of words to keep based on word frequency. Default is 10000.
    max_length (int): Maximum length for all sequences. If a sequence is shorter than the max length, 
                      it will be padded, if it is longer, it will be truncated. Default is 100.

    Returns
    ----------
    pd.DataFrame: A DataFrame with the tokenized and padded 'title' and 'text' columns.
    """
    # Initialize tokenizer
    tokenizer = Tokenizer(num_words=vocab_size, filters='')

    # Extract columns
    titles = df[title_column]
    texts = df[text_column]

    # Fit the tokenizer on the texts
    tokenizer.fit_on_texts(pd.concat([titles, texts]))

    # Convert texts to sequences
    sequences_titles = tokenizer.texts_to_sequences(titles)
    sequences_texts = tokenizer.texts_to_sequences(texts)

    # Pad sequences
    padded_titles = pad_sequences(sequences_titles, maxlen=max_length, padding='post')
    padded_texts = pad_sequences(sequences_texts, maxlen=max_length, padding='post')

    # Create new DataFrame with tokenized and padded sequences
    df_tokenized = df.copy()
    df_tokenized[title_column] = list(padded_titles)
    df_tokenized[text_column] = list(padded_texts)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    return df_tokenized


def tokenize_transformer(df):
    """
    Tokenizes the text data in a DataFrame and converts it to a TensorFlow Dataset.

    This function combines the 'title' and 'text' fields into a single string,
    tokenizes this string, and creates a TensorFlow Dataset where each item is a tuple
    containing the tokenized text and the corresponding label. The function uses the
    'bert-base-uncased' tokenizer and does not remove punctuation from the text.

    Parameters
    ----------
    df (pandas.DataFrame): The DataFrame containing the data. It should have three columns:
                           'title', 'text', and 'label'.

    Returns
    ----------
    tensorflow.data.Dataset: A TensorFlow Dataset where each item is a tuple containing the 
                             tokenized text and the corresponding label.
    """
    # initialize the tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # concatenate the 'title' and 'text' columns
    texts = df['title'].str.cat(df['text'], sep=' ')
    
    # tokenize the texts
    encodings = tokenizer(texts.tolist(), truncation=True, padding=True, max_length=512)

    # convert labels to tensorflow tensors
    labels = tf.convert_to_tensor(df['label'].tolist())

    # convert encodings to tensorflow datasets
    dataset = tf.data.Dataset.from_tensor_slices((
        dict(encodings),
        labels
    ))

    return dataset