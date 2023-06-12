import get_data
import lowercasetext
import find_and_replace
import preprocessing_space
import pandas as pd

def DEFCON5(NN=True,verbose=False,save_to_file=False,filename='DEFCON5_data'):
    """
    This function works like get_data.get_text_data() and also conducts the following operations:
    lowercase text, preprocess_space, find and replace (@#*1), remove stopwords, and the data cleaning included in get_data.
    
    Parameters
    None
    
    Kwargs
    NN=True: bool. Use if the output is to be used for Neural networks. If false, then stopwords will be removed
    verbose=False: bool. Set true if progress status is to be printed
    save_to_file=False: bool. If true, results are saved to a .csv file with the same format as the original dataset (WELFake_Dataset.csv).
    filename='DEFCON5_data.csv': string. Filename for saved file. Irrelevant if save_to_file=False. Do not include file extensions
    
    """
    if NN:
        typestring = 'Neural Network'
        filestring = '_NN.csv'
    else:
        typestring = 'Decision Tree Based Algorithm'
        filestring = '_DT.csv'
    if verbose:
        print('DEFCON5 data import for '+typestring+'\nGetting data...')
    X_text,y = get_data.get_text_data()
    X_title,y = get_data.get_title_data()
    
    if verbose:
        print('Making text lowercase...')
    X_text_lowercase = lowercasetext.Lowercase(X_text)
    X_title_lowercase = lowercasetext.Lowercase(X_title)
    
    if verbose:
        print('Converting "f*ck" to "wordwithasterisk"...')
    X_text_lwr_asterisk = find_and_replace.Replace_words(X_text_lowercase,identifier='\*',replace_with='wordwithasterisk')
    X_title_lwr_asterisk = find_and_replace.Replace_words(X_title_lowercase,identifier='\*',replace_with='wordwithasterisk')
    
    if verbose:
        print('Converting "#YOLO" to "wordwithhashtag"...')
    X_text_lwr_ast_hashtag = find_and_replace.Replace_words(X_text_lwr_asterisk,identifier='#',replace_with='wordwithhashtag')
    X_title_lwr_ast_hashtag = find_and_replace.Replace_words(X_title_lwr_asterisk,identifier='#',replace_with='wordwithhashtag')
    
    if verbose:
        print('Converting "@realdonaldtrump" to "wordwithat"...')
    X_text_lwr_ast_tag_at = find_and_replace.Replace_words(X_text_lwr_ast_hashtag,identifier='@',replace_with='wordwithat')
    X_title_lwr_ast_tag_at = find_and_replace.Replace_words(X_title_lwr_ast_hashtag,identifier='@',replace_with='wordwithat')
    
    if verbose:
        print('Converting "$123,456.789lightyears" to "wordwithnumber" (in text only)...')
    X_text_lwr_ast_tag_number = find_and_replace.Replace_words(X_text_lwr_ast_tag_at,identifier='0|1|2|3|4|5|6|7|8|9',replace_with='wordwithnumber')
    #X_title_lwr_ast_tag_number = find_and_replace.Replace_words(X_title_lwr_ast_tag_at,identifier='0|1|2|3|4|5|6|7|8|9',replace_with='wordwithnumber')
    X_title_lwr_ast_tag_number = X_title_lwr_ast_tag_at
    
    if verbose:
        print('Converting ""dickhead"" to "" dickhead ""...')
    X_text_lwr_ast_tag_num_spaced = preprocessing_space.preprocess_space(X_text_lwr_ast_tag_number)
    X_title_lwr_ast_tag_num_spaced = preprocessing_space.preprocess_space(X_title_lwr_ast_tag_number)
    
    X_text_final = X_text_lwr_ast_tag_num_spaced
    X_title_final = X_title_lwr_ast_tag_num_spaced
    
    if not NN:
        if verbose:
            print('Removing stopwords...')
        X_text_lwr_ast_tag_num_spa_stop = preprocessing_space.remove_stopwords(X_text_lwr_ast_tag_num_spaced)
        X_title_lwr_ast_tag_num_spa_stop = preprocessing_space.remove_stopwords(X_title_lwr_ast_tag_num_spaced)
        
        X_text_final = X_text_lwr_ast_tag_num_spa_stop
        X_title_final = X_title_lwr_ast_tag_num_spa_stop
    
    if save_to_file:
        if verbose:
            print('Saving to file...')
        data_to_export=pd.concat({'title': X_title_final,
                      'text': X_text_final,
                      'label': y},axis=1)
        data_to_export.to_csv(filename+filestring)
    
    if verbose:
        print('DEFCON5 done')
    
    return X_title_final, X_text_final, y

def DEFCON4(NN=True,verbose=False,save_to_file=False,filename='DEFCON4_data'):
    """
    This function works like get_data.get_text_data() and also conducts the following operations:
    lowercase text, preprocess_space, remove stopwords, and the data cleaning included in get_data.
    
    Parameters
    None
    
    Kwargs
    NN=True: bool. Use if the output is to be used for Neural networks. If false, then stopwords will be removed
    verbose=False: bool. Set true if progress status is to be printed
    save_to_file=False: bool. If true, results are saved to a .csv file with the same format as the original dataset (WELFake_Dataset.csv).
    filename='DEFCON5_data.csv': string. Filename for saved file. Irrelevant if save_to_file=False. Do not include file extensions
    
    """
    if NN:
        typestring = 'Neural Network'
        filestring = '_NN.csv'
    else:
        typestring = 'Decision Tree Based Algorithm'
        filestring = '_DT.csv'
    if verbose:
        print('DEFCON5 data import for '+typestring+'\nGetting data...')
    X_text,y = get_data.get_text_data()
    X_title,y = get_data.get_title_data()
    
    if verbose:
        print('Making text lowercase...')
    X_text_lowercase = lowercasetext.Lowercase(X_text)
    X_title_lowercase = lowercasetext.Lowercase(X_title)
        
    if verbose:
        print('Converting ""dickhead"" to "" dickhead ""...')
    X_text_lwr_spaced = preprocessing_space.preprocess_space(X_text_lowercase)
    X_title_lwr_spaced = preprocessing_space.preprocess_space(X_title_lowercase)
    
    X_text_final = X_text_lwr_spaced
    X_title_final = X_title_lwr_spaced
    
    if not NN:
        if verbose:
            print('Removing stopwords...')
        X_text_lwr_spa_stop = preprocessing_space.remove_stopwords(X_text_lwr_spaced)
        X_title_lwr_spa_stop = preprocessing_space.remove_stopwords(X_title_lwr_spaced)
        
        X_text_final = X_text_lwr_spa_stop
        X_title_final = X_title_lwr_spa_stop
    
    if save_to_file:
        if verbose:
            print('Saving to file...')
        data_to_export=pd.concat({'title': X_title_final,
                      'text': X_text_final,
                      'label': y},axis=1)
        data_to_export.to_csv(filename+filestring)
    
    if verbose:
        print('DEFCON4 done')
    
    return X_title_final, X_text_final, y

def DEFCON3(NN=True,verbose=False,save_to_file=False,filename='DEFCON3_data'):
    """
    This function works like get_data.get_text_data() and also conducts the following operations:
    lowercase text, preprocess_space, remove stopwords, and converting NaNs to empty strings
    
    Parameters
    None
    
    Kwargs
    NN=True: bool. Use if the output is to be used for Neural networks. If false, then stopwords will be removed
    verbose=False: bool. Set true if progress status is to be printed
    save_to_file=False: bool. If true, results are saved to a .csv file with the same format as the original dataset (WELFake_Dataset.csv).
    filename='DEFCON5_data.csv': string. Filename for saved file. Irrelevant if save_to_file=False. Do not include file extensions
    
    """
    if NN:
        typestring = 'Neural Network'
        filestring = '_NN.csv'
    else:
        typestring = 'Decision Tree Based Algorithm'
        filestring = '_DT.csv'
    if verbose:
        print('DEFCON5 data import for '+typestring+'\nGetting data...')
    df = pd.read_csv('WELFake_Dataset.csv')
    # Replace NaNs with empty string in 'title' and 'text' columns
    df['title'] = df['title'].fillna('')
    df['text'] = df['text'].fillna('')
    X_title = df['title']
    X_text = df['text']
    y = df['label']
    
    if verbose:
        print('Making text lowercase...')
    X_text_lowercase = lowercasetext.Lowercase(X_text)
    X_title_lowercase = lowercasetext.Lowercase(X_title)
        
    if verbose:
        print('Converting ""dickhead"" to "" dickhead ""...')
    X_text_lwr_spaced = preprocessing_space.preprocess_space(X_text_lowercase)
    X_title_lwr_spaced = preprocessing_space.preprocess_space(X_title_lowercase)
    
    X_text_final = X_text_lwr_spaced
    X_title_final = X_title_lwr_spaced
    
    if not NN:
        if verbose:
            print('Removing stopwords...')
        X_text_lwr_spa_stop = preprocessing_space.remove_stopwords(X_text_lwr_spaced)
        X_title_lwr_spa_stop = preprocessing_space.remove_stopwords(X_title_lwr_spaced)
        
        X_text_final = X_text_lwr_spa_stop
        X_title_final = X_title_lwr_spa_stop
    
    if save_to_file:
        if verbose:
            print('Saving to file...')
        data_to_export=pd.concat({'title': X_title_final,
                      'text': X_text_final,
                      'label': y},axis=1)
        data_to_export.to_csv(filename+filestring)
    
    if verbose:
        print('DEFCON3 done')
    
    return X_title_final, X_text_final, y