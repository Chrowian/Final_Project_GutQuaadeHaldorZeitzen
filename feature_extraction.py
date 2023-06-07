import numpy as np
import pandas as pd
import get_data


def flag_vocab(X, substrings):
    """
    takes dataframe X and a list of substrings to be searched for
    Returns a merged dataframe with X and 1/0 columns named after the list of substrings
    """
    # search of specific phrases - not currently used
    substrings = substrings
    column_names = X.columns
    bool_matrix = np.zeros((len(X[column_names[0]]), len(substrings)), dtype=int)

    def search_substrings(row, index):
        for j, substring in enumerate(substrings):
            for column_name in column_names:
                if substring in row[column_name]:
                    bool_matrix[index, j] = 1

    # Apply the function to each row of the DataFrame
    X.apply(lambda row: search_substrings(row, X.index.get_loc(row.name)), axis=1)
    result_df = pd.DataFrame(bool_matrix, columns=substrings, index=X.index)
    merged_df = X.join(result_df)

    return merged_df


if __name__ == "__main__":
    None

# final_df = flag_vocab(X, ['http', 'Trump', 'BREAKING'])
# print(final_df)

