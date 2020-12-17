import amendements_analysis.settings.base as stg

import amendements_analysis.infrastructure.building_dataset as build_data
import amendements_analysis.infrastructure.sentence_encoding as encode

import amendements_analysis.domain.clusters_finder as cluster
import amendements_analysis.domain.topic_finder as topic_finder
import amendements_analysis.domain.topic_predicter as topic_predicter

print('=============Preparing download of data...==============')

builder = build_data.DatasetBuilder(is_split_sentence=False, rewrite_csv=False)
df = builder.data
print('===========Dataset is loaded and ready to be used================')

cleaner = encode.DatasetCleaner(df, partition = 2)
sentences = cleaner.sentences
df_bert = cleaner.df_bert
encoder = encode.TextEncoder(sentences, finetuned_bert = True, batch_size=16)
sentence_embeddings = encoder.sentence_embeddings
print('===========Sentences of the dataset have been encoded according to camemBERT================')

cluster, umap_model = cluster.clusters_finder(sentence_embeddings, n_clusters = 13).models
print('===========Cluster & UMAP fit models from camemBERT embedding available================')

cleaner_lemmatizer = topic_finder.TextCleaner(flag_dict=stg.FLAG_DICT)
df_cleaned = cleaner_lemmatizer.transform(df_bert)
df_cleaned = df_cleaned.assign(Topic = cluster.labels_)
print('===========dataframe used for cluster is cleaned & lemmatized to find important words================')

word_finder = topic_finder.TopicWordsFinder(df_cleaned)
top_n_words, topic_sizes = word_finder.words_per_topic
for i in range(0,(len(topic_sizes))):
    print(i, top_n_wors[i][:10])
print(topic_sizes.head(13))
print('===========From topic top words, decide to attribute a topic for each topic number================')

amendement = ['Il faut absolument séparer le corps médical des docteurs avec le corps des infirmers pour garantir\
               la stabilité de notre régime de santé']
sentence_embedding = encoder(amendement)
prediction = topic_predicter.topic_predicter(sentence_embedding, umap_model, cluster).predicted_topic
print('prediction is : ', prediction)
print('corresponding to these words : ', top_n_wors[i][:10])
