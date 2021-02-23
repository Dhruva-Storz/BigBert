#!pip install pytorch-pretrained-bert
import torch
#from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
# If we use Bert from tranformers it outputs the same shape embedding as the 
# german one.
# Helpful : https://huggingface.co/transformers/model_doc/bert.html
from transformers import AutoModel, AutoTokenizer, BertTokenizer, BertModel
#
from preprocess import Preprocess


class Embedding():
    
    '''
    Parameters: 
        lang: 'en'/'english' or 'de'/'german'
        model: Pretrained Bert model if lang=='en' 
                Pretrained German Bert model if lang=='de'
        tokenizer: Pretrained Bert tokenizer if lang=='en' 
                Pretrained German Bert tokenizer if lang=='de'
        preprocesssor: Sentence preprocessor {Lemmatization/ Stop words removal}
    '''
    
    def __init__(self, lang):
        self.model = None
        self.tokenizer = None
        self.preprocessor = Preprocess(lang)
        self.lang = self._set_language(lang)
    
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
        'english' or 'german'
            DESCRIPTION. Sets the Encoder and Tokenizer.
        
        '''
        
        if lang == 'en' or lang == 'english':
            print('Loading English BERT tokenizer...\n')
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            print('Loading English BERT Model...\n')
            self.model = BertModel.from_pretrained('bert-base-uncased')
            
            return 'english'
            
        elif lang == 'de'or lang == 'german':
            print('Loading German BERT tokenizer...\n')
            self.tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-cased")
            print('Loading German BERT Model...\n')
            self.model = AutoModel.from_pretrained("dbmdz/bert-base-german-cased")
            
            return 'german'

        else:
            raise ValueError('{} {}'.format("Input must be one of two: 'en' or",
                             "english' for English, 'de' or 'german' for German."))
            
    def get_batch_embedding(self, sentences, batch_size, lemmatize=False, remove_stop_words=False):
        '''
        Parameters
        ----------
        sentences : list
            DESCRIPTION. A list of raw text sentences.
        batch_size : int
            DESCRIPTION. Sentences are split into sets of length batch_size 
                        to be fed into the model.
        lemmatize : TYPE, optional
            DESCRIPTION. The default is False. If True the preprocessor is used to
                        lemmatise the sentence. 
        remove_stop_words : bool, optional
            DESCRIPTION. The default is False. If True the preprocessor is used to
                        remove the stop-words in each sentence.                       

        Returns
        -------
        embedding_max : Tensor of shape (len(sentences), 768)

        embedding_avg : Tensor of shape (len(sentences), 768)

        '''
        #https://mccormickml.com/2019/07/22/BERT-fine-tuning/
        embedding_max = torch.empty(len(sentences), 768)
        embedding_avg = torch.empty(len(sentences), 768)
        for b in range(0, len(sentences), batch_size):
            print('Processing batch {}...'.format(int(b / batch_size + 1)))
            batch = sentences[b:b+batch_size]
            
            tokenized_sentences = []
            # Preprocess and tokenize batch
            for sentence in batch:
                sentence = " ".join([word if word[-3:] != "n't" else word + " not" for word in sentence.split()])
                if lemmatize:
                    sentence = self.preprocessor.get_preprocessed_sentence(sentence, remove_stop_words)
                tokenized_sentence = self.get_tokenized_sentence(sentence)
                tokenized_sentences.append(tokenized_sentence)
            current_batch_size = len(tokenized_sentences)
            max_length = len(max(tokenized_sentences, key=len))
            
            # Create tokens_tensor/segmentes_tensor of length (current_batch_size, max_length)
            # index 0 in the vocab = "[PAD]"
            # zeros in tokens_tensor are equal to [PAD]
            # zeros in segments_tensor correspond to padded words
            
            tokens_tensor = torch.zeros(current_batch_size, max_length, dtype=torch.long)
            segments_tensor = torch.zeros(current_batch_size, max_length, dtype=torch.long)
            for i in range(len(tokenized_sentences)):
                sent = tokenized_sentences[i]
                sent_len = len(sent)
                
                tokens_tensor[i,:sent_len] = self.get_token_tensor(
                    self.get_indexed_tokens(sent)
                    )
                segments_tensor[i,:sent_len] = self.get_segments_tensor(
                    self.get_segment_indices(sent)
                    )
            # Put the model to evaluation mode
            self.model.eval()
            with torch.no_grad():
                print('Getting embedding for batch {}...\n'.format(int(b / batch_size + 1)))
                encoded_layers, _ = self.model(tokens_tensor, segments_tensor)
            
            # Average or take the max embeddings with respect to dim=1
            batch_embedding_max, _ = encoded_layers.max(dim=1)
            batch_embedding_avg = encoded_layers.mean(dim=1)
        
            embedding_max[b:b+batch_size,:] = batch_embedding_max
            embedding_avg[b:b+batch_size,:] = batch_embedding_avg
        print('DONE!')
        print('Embedding size is ({},{})'.format(embedding_max.shape[0], embedding_max.shape[1]))
        return embedding_max, embedding_avg

    def get_tokenized_sentence(self, preprocessed_sentence):
        '''
        Parameters
        ----------
        preprocessed_sentence : str
            DESCRIPTION. A preprocessed sentence to be tokenized. 
                    [CLS] marks the beggining and [SEP] the ending of the sentence.

        Returns
        -------
        tokenized_sentence : list
            DESCRIPTION. A list with tokens of the sentence.

        '''
        # "[CLS]" marks the beggining of a sentence
        # "[SEP]" marks the ending of a sentence
        marked_sentence = "[CLS] " + preprocessed_sentence + " [SEP]"
        tokenized_sentence = self.tokenizer.tokenize(marked_sentence)
        
        return tokenized_sentence
    
    def get_indexed_tokens(self, tokenized_sentence):
        
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_sentence)
        
        return indexed_tokens

    def get_segment_indices(self, tokenized_sentence):
        
        return [1] * len(tokenized_sentence)
    
    def get_token_tensor(self, indexed_tokens):
        '''
        To be called with get_indexed_tokens
        '''
        
        tokens_tensor = torch.tensor([indexed_tokens])
        
        return tokens_tensor
    
    def get_segments_tensor(self, segments_ids):
        '''
        To be called with get_segment_indices
        '''
        
        segments_tensors = torch.tensor([segments_ids])
        
        return segments_tensors
        
        