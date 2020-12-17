import amendements_analysis.settings.base as stg
import numpy as np
import pandas as pd
import os 
import torch
from transformers import AutoTokenizer, CamembertTokenizer
from transformers import CamembertModel
from tqdm import tqdm



class DatasetCleaner():
    """
    Get amendments df and perform pre processing compatible with
    -  BERT embedding stence transformers

    Attributes
    ----------
    df: pandas.DataFrame
    partition: int
        choose to return size(df) / partition

    Properties
    ----------
    data: pandas.DataFrame
    """

    def __init__(self, df, partition=1):
        """Class initilization
        Parameters
        ----------
        df: pandas.DataFrame
        partition: int
            choose to return size(df) / partition
        """
        self.partition = partition
        self.df = df
        self.df_bert = self.get_df_bert()
        
    def get_df_bert(self):
        df_bert = self.df.copy()
        df_bert = df_bert[df_bert[stg.AMENDEMENT].notna()]
        df_bert[stg.AMENDEMENT] = df_bert[stg.AMENDEMENT].astype(str)
        df_bert.drop_duplicates(subset=stg.AMENDEMENT,keep = False, inplace = True)
        return df_bert.iloc[:: self.partition]
    
    @property
    def sentences(self):
        sentences = self.df_bert[stg.AMENDEMENT].to_list()
        return sentences

class TextEncoder():
    """
    Get amendments list and perform BERT embedding with: 
    -  french camemBERT 
    -  french camemBERT fine tuned with our text corpus

    Attributes
    ----------
    sentences: list of strings to encode by Bert
    custom: bool, optionnal
        if true, use custom finetuned camemBERT 

    Return
    ----------
    vector: np.array
    """
    def __init__(self, sentences, finetuned_bert= False, batch_size = 8):
        """Class initilization

        Parameters
        ----------
        sentences: list of strings to encode by Bert
        custom: bool, optionnal
            if true, use custom finetuned camemBERT 
        """

        self.sentences = sentences
        self.finetuned_bert = finetuned_bert
        self.batch_size = batch_size

    @property
    def sentence_embeddings(self):
        if self.finetuned_bert == False:
            tokenizer = CamembertTokenizer.from_pretrained(stg.REGULAR_CAMEMBERT)
            model = CamembertModel.from_pretrained(stg.REGULAR_CAMEMBERT)
        else:
            tokenizer = AutoTokenizer.from_pretrained(stg.FINED_TUNED_CAMEMBERT)
            model = CamembertModel.from_pretrained(stg.FINED_TUNED_CAMEMBERT)
        if torch.cuda.is_available() == True :
            print('====== Cuda is Available, GPU will be used for this task ======')
            model.cuda()
            device = torch.device("cuda")
        embedding_all_text=[]
        number_sentences = len(self.sentences)
        for i in tqdm(range(0,number_sentences, self.batch_size)):
            if ((i+self.batch_size) < number_sentences):
                batch = self.sentences[i:i+self.batch_size]
                encoded_input = self.get_batch_sentence_tokens(batch, tokenizer)
            elif (i == number_sentences):
                pass
            else :
                batch = sentences[i:]
                encoded_input = self.get_batch_sentence_tokens(batch, tokenizer)
            if torch.cuda.is_available() == True :
                encoded_input.to(device)
            model_output = model(**encoded_input)
            sentence_embeddings_tensor = self.mean_pooling(model_output, encoded_input['attention_mask'])
            embedding_all_text.append(sentence_embeddings_tensor) 
            if torch.cuda.is_available() == True:
                torch.cuda.empty_cache()
        sentence_embeddings = self.torch_to_array(embedding_all_text)
        return sentence_embeddings 
    
    def get_batch_sentence_tokens(self, batch, tokenizer):
        encoded_input = tokenizer(batch, 
                                  padding=True, 
                                  truncation=True, 
                                  max_length=512, 
                                  return_tensors='pt')
        return encoded_input 
        
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def torch_to_array(self, tensor_sentence_embeddings):
        if torch.cuda.is_available() == True:
            tensor_sentence_embeddings = [x.cpu() for x in tensor_sentence_embeddings]
        sentence_embeddings = [x.detach().numpy() for x in tensor_sentence_embeddings]
        sentence_embeddings = np.concatenate(sentence_embeddings)
        return sentence_embeddings

if __name__ == "__main__":
    df = pd.read_csv(os.path.join(stg.DATA_DIR, "amendements_sentence.csv"))
    sentences = DatasetCleaner(df, partition = 20000).sentences
    df_bert = DatasetCleaner(df, partition = 20000).df_bert
    sentence_embeddings = TextEncoder(sentences, finetuned_bert = True, batch_size=2).sentence_embeddings
