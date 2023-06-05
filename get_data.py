import pandas as pd

def clean_data():
    """
    Removes all non-english articles, cleans typesetting artifacts and removes articles with unusually low or high average word length
    :return: pandas dataframs containing title, text and label
    """
    df = pd.read_csv('WELFake_Dataset.csv')
    df[['title', 'text']] += " "
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    df.replace(to_replace='', value=pd.NA, inplace=True)
    df.dropna(inplace=True)

    # general cleanup of typesetting artifacts, \n ect...
    df['text'] = df['text'].str.replace(r'\n', '', regex=True)  # removes any \n
    df['text'] = df['text'].str.replace('  ', ' ')
    df['title'] = df['title'].str.replace('  ', ' ')
    # remove non english datapoints
    pattern = r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFBCF\uFE70-\uFEFF\u0590-\u05FF\u4E00-\u9FFF]+'
    non_english_mask = df['text'].astype(str).str.contains(pattern, regex=True)
    df_new = df[~non_english_mask.values]

    """
    # search of specific phrases - not currently used
    substrings = ['http',]
    for idx, substring in enumerate(substrings):
        if idx == 0:
            is_present = df_new.astype(str).apply(lambda x: x.str.contains(substring)).any(axis=1)  # mask where https occurs in either title or text
        else:
            is_present += df_new.astype(str).apply(lambda x: x.str.contains(substring)).any(axis=1)  # mask where https occurs in either title or text

    data_list = np.arange(0, len(df_new['text']), 1)
    data_list = data_list[is_present]
    """

    whitespace_cnt = df_new['text'].str.count(' ')
    char_cnt = df_new['text'].str.len()
    avg_word_len = (char_cnt-whitespace_cnt) / whitespace_cnt

    final_drop_idx = avg_word_len.index.to_series()[avg_word_len < 3]
    final_drop_idx_2 = avg_word_len.index.to_series()[avg_word_len > 10]
    df_new.drop(final_drop_idx)
    df_new.drop(final_drop_idx_2)
    return df_new


def get_all_data():
    """
    :return: pandas dataframe X (title and text) and y (label)
    """
    df = clean_data()
    X = df[['title', 'text']]
    y = df['label']
    print('X shape: ', X.shape)
    print('y shape: ', y.shape)
    return X, y


def get_title_data():
    """
    :return: pandas dataframe X (title) and y (label)
    """
    df = clean_data()
    X_title = df['title']
    y = df['label']
    print('X shape: ', X_title.shape)
    print('y shape: ', y.shape)
    return X_title, y


def get_text_data():
    """
    :return: pandas dataframe X (text) and y (label)
    """
    df = clean_data()
    X_text = df['text']
    y = df['label']
    print('X shape: ', X_text.shape)
    print('y shape: ', y.shape)
    return X_text, y

X, y = get_all_data()