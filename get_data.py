import pandas as pd


def get_all_data():
    """
    :return: pandas dataframe X (title and text) and y (label)
    """
    df = pd.read_csv('WELFake_Dataset.csv')
    df.dropna(inplace=True)
    X = df[['title', 'text']]
    X += " "
    y = df['label']
    print('X shape: ', X.shape)
    print('y shape: ', y.shape)
    return X, y


def get_title_data():
    """
    :return: pandas dataframe X (title) and y (label)
    """
    df = pd.read_csv('WELFake_Dataset.csv')
    df.dropna(inplace=True, subset=['title'])
    X_title = df['title']
    X_title += " "

    y = df['label']
    print('X shape: ', X_title.shape)
    print('y shape: ', y.shape)
    return X_title, y


def get_text_data():
    """
    :return: pandas dataframe X (text) and y (label)
    """
    df = pd.read_csv('WELFake_Dataset.csv')
    df.dropna(inplace=True, subset=['text'])
    X_text = df['text']
    X_text += " "
    y = df['label']
    print('X shape: ', X_text.shape)
    print('y shape: ', y.shape)
    return X_text, y


if __name__ == "__main__":
    None


