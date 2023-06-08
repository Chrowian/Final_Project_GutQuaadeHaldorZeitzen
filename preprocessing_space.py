import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy.sparse import csr_matrix, hstack

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



from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer

# for tree based classifiers

def preprocess_text(df, text_columns, non_text_columns=None, max_features=40000, print_vocabulary=False):
    """
    Function to preprocess text columns and combine with other features in a DataFrame.

    Args
    ----------
        df (pandas.DataFrame): the input DataFrame.

        text_columns (list of str): the names of the text columns to vectorize. ['title', 'text']

        non_text_columns (list of str): the names of the non-text columns to include in the feature matrix. Defaults to None.

        max_features (int, optional): the maximum number of features for the TfidfVectorizer. Defaults to 5000.

        print_vocabulary (bool, optional): if True, print the vocabulary of each text column. Defaults to False.

    Returns
    ----------
        X (scipy.sparse.csr.csr_matrix): the combined feature matrix.

        vectorizers (dict): dictionary mapping column names to trained TfidfVectorizers.
    """

    vectorizers = {}
    feature_matrices = []

    for col in text_columns:
        vectorizer = TfidfVectorizer(max_features=max_features)
        X_col = vectorizer.fit_transform(df[col])
        vectorizers[col] = vectorizer
        feature_matrices.append(X_col)
        
        # Print the vocabulary if requested
        if print_vocabulary:
            print(f'Vocabulary for {col}: {vectorizer.get_feature_names_out()}')

    if non_text_columns is not None:
        for col in non_text_columns:
            X_col = csr_matrix(df[col].values).T  # Transpose to get a column vector
            feature_matrices.append(X_col)

    X = hstack(feature_matrices)

    return X, vectorizers



# def preprocess_text_old(df, columns, use_stemming=True, max_features=5000):
#     """
#     Preprocesses the text data for the given dataframe and columns using optional stemming and TF-IDF vectorization.
    
#     Parameters
#     ----------
#     df : pandas.DataFrame
#         The dataframe containing the text data.
#     columns : list of str
#         The columns in the dataframe to preprocess.
#     use_stemming : bool, default=True
#         If True, use stemming during the preprocessing.
#     max_features : int, default=5000
#         Maximum number of features for the TF-IDF vectorizer (i.e., maximum vocabulary size).

#     Returns
#     -------
#     df_transformed : pandas.DataFrame
#         The dataframe with the original non-text columns and the transformed text columns.
#     vectorizers : dict
#         A dictionary mapping column names to their corresponding TF-IDF vectorizers.
#     """
#     # Initialize stemmer
#     stemmer = PorterStemmer()

#     # Define the stemming function
#     def stem_text(text):
#         words = word_tokenize(text)
#         words = [stemmer.stem(word) for word in words]
#         return ' '.join(words)

#     df_transformed = df.copy()
#     vectorizers = {}

#     for col in columns:
#         if use_stemming:
#             df_transformed[col] = df_transformed[col].apply(stem_text)

#         # Initialize the vectorizer
#         vectorizer = TfidfVectorizer(max_features=max_features)
        
#         # Fit and transform the vectorizer on our text
#         df_transformed[col] = list(vectorizer.fit_transform(df_transformed[col]).toarray())
        
#         # Store the vectorizer
#         vectorizers[col] = vectorizer

#     return df_transformed, vectorizers