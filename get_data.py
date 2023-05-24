import pandas as pd


if __name__ == "__main__":
    pass


def get_data():
    """

    :return: pandas dataframe X and y
    """
    df = pd.read_csv('WELFake_Dataset.csv')
    X = df[['title', 'text']]
    y = df['label']
    print('X shape: ', X.shape)
    print('y shape: ', y.shape)
    return X, y
