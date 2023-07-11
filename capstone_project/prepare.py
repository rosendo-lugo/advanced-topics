#standard imports
import pandas as pd
import numpy as np

# my imports
import acquire as a

#import Parsing Text
import unicodedata

#import regular expression operations
import re

#import natural language toolkit
import nltk

#import stopwords list
from nltk.corpus import stopwords

'''
*------------------*
|                  |
|     PREPARE      |
|                  |
*------------------*
'''

# ------------------------------------------------------------------------------------
def basic_clean(string):
    """
    Lower Case:
    - setting all letters to a lowercase
    
    Encoding:
    - `unicodedata.normalize` removes any inconsistencies in unicode character encoding
    - `.encode` to convert the resulting string to the ASCII character set
    - `.decode` to turn the resulting bytes object back into a string
    
    Special characters:
    - remove anything that isn't a-z, a number, a single quote, or a whitespace
    """
    # lowercase text
    string = string.lower()
    
    # remove any accented characters and non-ASCII characters
    # normalizing
    # getting ride of anything not in ascii
    # turning back to a string
    string = unicodedata.normalize('NFKD', string).encode('ascii','ignore').decode('utf-8')
    
    # remove special characters
    #use re.sub to remove special characters
    bc_string = re.sub(r'[^a-z0-9\'\s]', '', string)
    
    return bc_string
# ------------------------------------------------------------------------------------
def tokenize(string):
    """
    Tokenization is the process of breaking something down
    into smaller, discrete units. These units are called tokens.
    
    It's common to tokenize the strings to break up words and punctutation
    left over into discrete units. 
    """  

    #create the tokenizer
    tokenize = nltk.tokenize.ToktokTokenizer()
    tok_string = tokenize.tokenize(string, return_str=True)
  
    return tok_string

# ------------------------------------------------------------------------------------
def stem(string):
    """
    Stemming:
    - **truncates** words to their "stem"
    - algorithmic rules (non lingustic)
    - example: "calls", "called", "calling" --> "call"
    - fast and efficient
    """   
    #create porter stemmer
    ps = nltk.porter.PorterStemmer()
    
    #use stemmer - apply stem to each word in our string
    ps.stem(string)
    
    # split all the words in the article
    string.split()
    stems = [ps.stem (word) for word in string.split()]
    
    #join words back together
    string_stemmed = ' '.join(stems)
    
    return string_stemmed

# ------------------------------------------------------------------------------------
def lemmatize(string):
    """
    Lemmatize:
        - **changes** words to their "root"
        - it can conjugate to the base word 
        - example: "mouse", "mice" --> "mouse"
        - slower than stemming
    """ 
    #create the lemmatizer   
    wnl = nltk.stem.WordNetLemmatizer()
    
    #use lemmatize - apply stem to each word in our string
    # wnl.lemmatize(article)
    lemma = [wnl.lemmatize(word) for word in string.split()]
    
    #join words back together
    string_lemma = ' '.join(lemma)
    
    return string_lemma

# ------------------------------------------------------------------------------------
def remove_stopwords(string, string_lemma):
    """
    Words which have little or no significance, especially when constructing
    meaningful features from text, are known as stopwords.
    - example: a, an, the, and like

    We will use a standard English language stopwords list from nltk
    """
    #save stopwords
    stopwords_ls = stopwords.words('english')
    
    #split words in lemmatized article
    words = string_lemma.split()
    
    #remove stopwords from list of words
    filtered = [word for word in words if word not in stopwords_ls]
    
    #join words back together
    rem_stopwords = ' '.join(filtered)
    
    return rem_stopwords

# ------------------------------------------------------------------------------------
def remove_stopwords_extra_words(string_lemma, extra_words, exclude_words):
    """
    Words which have little or no significance, especially when constructing
    meaningful features from text, are known as stopwords.
    - example: a, an, the, and like

    We will use a standard English language stopwords list from nltk
    """
    #save stopwords
    stopwords_ls = stopwords.words('english')
    
    # remove extra words
    stopwords_ls = set(stopwords_ls) - set(exclude_words)

    # add to stopword list
    stopwords_ls = set(stopwords_ls).union(extra_words)
    
    #split words in lemmatized article
    words = string_lemma.split()
    
    #remove stopwords from list of words
    filtered = [word for word in words if word not in stopwords_ls]
    
    #join words back together
    rem_stopwords = ' '.join(filtered)
    
    return rem_stopwords

# ------------------------------------------------------------------------------------
ADDITIONAL_STOPWORDS = ['r', 'u', '2', '4', 'ltgt']

def clean(text):
    '''
    A simple function to cleanup text data.
    
    Args:
        text (str): The text to be cleaned.
        
    Returns:
        list: A list of lemmatized words after cleaning.
    '''
    
    # basic_clean() function from last lesson:
    # Normalize text by removing diacritics, encoding to ASCII, decoding to UTF-8, and converting to lowercase
    text = (unicodedata.normalize('NFKD', text)
             .encode('ascii', 'ignore')
             .decode('utf-8', 'ignore')
             .lower())
    
    # Remove punctuation, split text into words
    words = re.sub(r'[^\w\s]', '', text).split()
    
    
    # lemmatize() function from last lesson:
    # Initialize WordNet lemmatizer
    wnl = nltk.stem.WordNetLemmatizer()
    
    # Combine standard English stopwords with additional stopwords
    stopwords = nltk.corpus.stopwords.words('english') + ADDITIONAL_STOPWORDS
    
    # Lemmatize words and remove stopwords
    cleaned_words = [wnl.lemmatize(word) for word in words if word not in stopwords]
    
    return cleaned_words

# ------------------------------------------------------------------------------------
def clean(text: str) -> list: 
    """A simple function to cleanup text data"""
    
    #remove non-ascii characters & lower
    text = (text.encode('ascii', 'ignore')
                .decode('utf-8', 'ignore')
                .lower())
    
    #remove special characters
    words = re.sub(r'[^\w\s]', '', text).split()
    
    #build the lemmatizer
    wnl = nltk.stem.WordNetLemmatizer()
    
    #getting all stopwords
    stopwords = set(nltk.corpus.stopwords.words('english'))
    
    return [wnl.lemmatize(word) for word in words if word not in stopwords]

