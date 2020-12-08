import amendements_analysis.settings.base as stg
from sentence_transformers import SentenceTransformer, models
import numpy as np
import pandas as pd
import os 

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
        df_bert = df_bert.groupby(stg.AMENDEMENT)\
                         .first()\
                         .reset_index()\
                         .iloc[:: self.partition]
        return df_bert
    
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
    def __init__(self, sentences, finetuned_bert= False):
        """Class initilization

        Parameters
        ----------
        sentences: list of strings to encode by Bert
        custom: bool, optionnal
            if true, use custom finetuned camemBERT 
        """

        self.sentences = sentences
        self.finetuned_bert = finetuned_bert

    @property
    def sentence_embeddings(self):
        if self.finetuned_bert == False:
            word_embedding_model = models.CamemBERT('camembert-base')
            dim = word_embedding_model.get_word_embedding_dimension()
            pooling_model = models.Pooling(dim, 
                                           pooling_mode_mean_tokens=True, 
                                           pooling_mode_cls_token=False, 
                                           pooling_mode_max_tokens=False
                                           )
            model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
            sentence_embeddings = model.encode(self.sentences,show_progress_bar = True)

        else:
            pooling_model = models.Pooling(768, 
                                           pooling_mode_mean_tokens=True, 
                                           pooling_mode_cls_token=False, 
                                           pooling_mode_max_tokens=False
                                           )
            model = SentenceTransformer(modules=['./my/path/to/model/', pooling_model])

        return sentence_embeddings


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(stg.DATA_DIR, "amendements_sentence.csv"))
    sentences = DatasetCleaner(df, partition = 20000).sentences
    df_bert = DatasetCleaner(df, partition = 20000).df_bert
    sentence_embeddings = TextEncoder(sentences).sentence_embeddings
