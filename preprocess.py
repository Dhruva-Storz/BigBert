import spacy
from spacy.lang import en as english
from spacy.lang import de as german
# Using SPACY
stop_words_en = english.stop_words.STOP_WORDS
stop_words_de = german.stop_words.STOP_WORDS

##### Run on command prompt (only once)
#!python -m spacy download en_core_web_sm
#!python -m spacy download de_core_news_sm


import nltk
from nltk.tokenize import word_tokenize
#from nltk import download
#from nltk.corpus import stopwords
#download('stopwords','punkt')

# Using NLTK
#stop_words_en = set(stopwords.words('english'))
#stop_words_de = set(stopwords.words('german'))

class Preprocess():
    
    '''
    Parameters: 
        lang: 'en'/'english' or 'de'/'german'
        lemmatizer: spacy.load('en_core_web_sm').tokenizer if lang=='en' 
                spacy.load('de_core_web_sm').tokenizer if lang=='de'
        stop_words: spacy.lang.en.stop_words.STOP_WORDS if lang=='en' 
                spacy.lang.de.stop_words.STOP_WORDS if lang=='de'
        
    '''
    
    def __init__(self, lang='en'):
        self.stop_words = None
        self.lemmatizer = None
        self.lang = self._set_language(lang)
        self.word_tokenizer = word_tokenize
        
    
    def sentences_tokenize(self, document):
        if isinstance(document, str):
            return self.words_tokenize(document)
        
        if isinstance(document, list):
            if len(document) == 1:
                return self.words_tokenize(document[0])
            
            tokenized_doc = []
            for sentence in document:
                tokenized_doc.append(self.words_tokenize(sentence))
            
            return tokenized_doc
        else:
            raise TypeError('Not supported type of input.')
    
    def words_tokenize(self, sentence):
        sentence = self.remove_stopwords(sentence)
        sentence = self.lemmatize_sentence(sentence)
        
        return " ".join(self.word_tokenizer(sentence.lower()))
    
    def get_preprocessed_sentence(self, sentence, remove_stop_words=False):
        if remove_stop_words:
            sentence = self.remove_stopwords(sentence.lower())
            sentence = self.lemmatize_sentence(sentence)
        
        else:
            sentence = self.lemmatize_sentence(sentence.lower())
        
        return sentence
        
    
    def _set_language(self, lang):
        '''

        Parameters
        ----------
        lang : str
            DESCRIPTION. 'en' or 'english' for English
                        'de' or 'german' for German

        Raises
        ------
        ValueError
            DESCRIPTION. If input is not in {'en', 'english', 'de', 'german'}

        Returns
        -------
        lang : str 
            DESCRIPTION. 'english' or 'german'

        '''
        
        if lang == 'en' or lang == 'english':
            self.stop_words = english.stop_words.STOP_WORDS
            self.lemmatizer = spacy.load('en_core_web_sm').tokenizer
        elif lang == 'de'or lang == 'german':
            self.stop_words = german.stop_words.STOP_WORDS
            self.lemmatizer = spacy.load('de_core_news_sm').tokenizer
        else:
            raise ValueError('{} {}'.format("Input must be one of two: 'en' or",
                             "english' for English, 'de' or 'german' for German."))
            
        return lang
    def remove_stopwords(self, sentence):
        if isinstance(sentence, str):
            return " ".join([word for word in sentence.split() if word 
                             not in self.stop_words ])
        else:
            removed_stop_words_sent = []
            for sent in sentence:
                print(sent)
                removed_stop_words_sent.append(" ".join(
                    [word for word in sent.split() if word 
                             not in self.stop_words ]))
            return removed_stop_words_sent
                
    def lemmatize_sentence(self, sentence):
        lemmatized_sentence = []
        for word in sentence.split():
            lemmatized_sentence.append(self.lemmatizer(word.lower())[0].lemma_)
        return " ".join(lemmatized_sentence)
        
   