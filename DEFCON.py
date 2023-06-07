import get_data
import lowercasetext
import find_and_replace
import preprocessing_space

def DEFCON5(NN=True,verbose=False):
    """
    This function works like get_data.get_text_data() and also conducts the following operations:
    lowercase text
    
    Kwargs
    NN=True: bool. Use if the output is to be used for Neural networks. If false, then stopwords will be removed
    
    """
    if NN:
        typestring = 'Neural Network'
    else:
        typestring = 'Tree Based Algorithm'
    if verbose:
        print('DEFCON5 data import for '+typestring+'\nGetting data...')
    X_text,y = get_data.get_text_data()
    
    if verbose:
        print('Making text lowercase...')
    X_text_lowercase = lowercasetext.Lowercase(X_text)
    
    if verbose:
        print('Converting "f*ck" to "wordwithasterisk"...')
    X_text_lwr_asterisk = find_and_replace.Replace_words(X_text_lowercase,identifier='\*',replace_with='wordwithasterisk')
    
    if verbose:
        print('Converting "#YOLO" to "wordwithhashtag"...')
    X_text_lwr_ast_hashtag = find_and_replace.Replace_words(X_text_lwr_asterisk,identifier='#',replace_with='wordwithhashtag')
    
    if verbose:
        print('Converting "@realdonaldtrump" to "wordwithat"')
    X_text_lwr_ast_tag_at = find_and_replace.Replace_words(X_text,identifier='@',replace_with='wordwithat')
    
    if verbose:
        print('Converting "$123,456.789lightyears" to "wordwithnumber"...')
    X_text_lwr_ast_tag_number = find_and_replace.Replace_words(X_text_lwr_ast_tag_at,identifier='0|1|2|3|4|5|6|7|8|9',replace_with='wordwithnumber')
    
    if NN:
        return X_text_lwr_ast_tag_number
    else:
        if verbose:
            print('Removing stopwords...')
        X_text_lwr_ast_tag_num_stop = preprocessing_space.remove_stopwords(X_text_lwr_ast_tag_number)
        if verbose:
            print('DEFCON5 done...')
        return X_text_lwr_ast_tag_num_stop