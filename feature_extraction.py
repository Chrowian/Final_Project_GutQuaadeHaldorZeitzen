import numpy as np
import pandas as pd
import get_data





def calculate_entropy(fake_occ, real_occ, fake_non_occ, real_non_occ):
    total_fake = fake_occ + fake_non_occ
    total_real = real_occ + real_non_occ
    total = total_fake + total_real

    p_fake_with_word = fake_occ / total
    p_real_with_word = real_occ / total

    entropy_parent = -(p_fake_with_word * np.log2(p_fake_with_word) + p_real_with_word * np.log2(p_real_with_word))
    entropy_parent[np.isnan(entropy_parent)] = 0  # Set NaN values to 0

    return entropy_parent


def calculate_entropy_gain(fake_occ, real_occ, fake_non_occ, real_non_occ):
    entropy_parent = calculate_entropy(fake_occ, real_occ, fake_non_occ, real_non_occ)

    total_fake = fake_occ + fake_non_occ
    total_real = real_occ + real_non_occ
    total = total_fake + total_real

    p_fake = total_fake / total
    p_real = total_real / total

    entropy_children = (
        p_fake * calculate_entropy(fake_occ, fake_non_occ, fake_non_occ, real_non_occ) + p_real * calculate_entropy(real_occ, real_non_occ, fake_non_occ, real_non_occ)
    )
    entropy_children[np.isnan(entropy_children)] = 0  # Set NaN values to 0

    entropy_gain = entropy_parent - entropy_children
    return entropy_gain


def flag_vocab(X, y):
    """
    Just call flag vocab on X and y - returns X with 22 added columns: title and text length (word cnt) and 20
    additional words (in article or not), which gives the larges entropy gain when split as well as the list of words
    """
    # search of specific phrases - not currently used
    substrings = ['bizarre', 'discovery', 'legislation', 'legislative', 'cataclysmic', 'event', 'inauguration', 'unheard', 'earth-shattering', 'election', 'claim', 'bipartisan', 'unexplained', 'office', 'inside', 'senate', 'supreme', 'blockbuster', 'unveiled', 'exposed']
    bool_matrix = np.zeros((len(y), len(substrings)), dtype=int)

    def search_substrings(row, index):
        for j, substring in enumerate(substrings):
            # print(row.name)
            for column_name in X.columns:
                if substring in row[column_name]:
                    # if substring in row['title'] or substring in row['text']:
                    bool_matrix[index, j] = 1

    # Apply the function to each row of the DataFrame
    X.apply(lambda row: search_substrings(row, X.index.get_loc(row.name)), axis=1)

    def count_words(text):
        return len(text.split())

    # Apply the function to count words in each string
    X['word count text'] = X['text'].apply(lambda x: count_words(x))
    X['word count title'] = X['title'].apply(lambda x: count_words(x))

    result_df = pd.DataFrame(bool_matrix, columns=substrings, index=X.index)
    merged_df = X.join(result_df)

    return merged_df, substrings


X, y = get_data.get_all_data()
X_new, substrings = flag_vocab(X, y)
print(X_new)
print(substrings)

if __name__ == "__main__":
    None

# final_df = flag_vocab(X, ['http', 'Trump', 'BREAKING'])
# print(final_df)

