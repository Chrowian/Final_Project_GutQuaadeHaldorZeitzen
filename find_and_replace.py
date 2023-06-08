import pandas as pd
import numpy as np
import re

def replace_words(input_string,identifier='\*',replace_with='wordwithasterisk'):
    """
    Finds words in 'input_string' containing 'identifier' and replaces them with 'replace_with' and returns the result
    This function has a small bug: If the last character in input_string is a space (' '), then that space will not be included in the output string.
    Function returns error if input_string is empty
    
    Parameters
    input_string: string
    
    Kwargs
    identifier: string
    replace_with: string
    """
    #split_by_identifier2=input_string.split(identifier)
    split_by_identifier=re.split(identifier,input_string)
    empty_strings_removed = list(filter(lambda x: len(x)>0, split_by_identifier))
    output_text=empty_strings_removed[0].rsplit(' ',1)[0]
    for i in range(len(empty_strings_removed)-2):
        i+=1
        text_addition=empty_strings_removed[i].rsplit(' ',1)[0]
        text_addition=text_addition.split(' ',1)[-1]
        output_text+=' '+replace_with+' '+text_addition
    if (len(empty_strings_removed)) > 1:
        output_text+=' '+replace_with+' '+empty_strings_removed[-1].split(' ',1)[-1]
    return output_text

def Replace_words(data,identifier='\*',replace_with='wordwithasterisk'):
    """
    Applies replace_words to all entries in data. kwargs are passed to replace_words.
    Returns data in the same format as it was given (dataframe or series)
    
    Parameters
    data: pandas dataframe or series
    
    Kwargs
    identifier: string
    replace_with: string   
    """
    kwargs = {'identifier': identifier, 'replace_with': replace_with}
    try:
        return data.apply(np.vectorize(replace_words),**kwargs)
    except:
        return data.applymap(replace_words,**kwargs)