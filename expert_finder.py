import pandas as pd
import numpy as np
from ast import literal_eval
from transformers import BertTokenizer, BertModel
import networkx as nx
from src.controller import generator, trainer
from src.lib import extractor

DATA_PATH = './data/'
MODEL_PATH = './model/'

def prepare_data():
    ''' This function reads data from the existing data source'''
    global DATA_PATH
    
    doc_df = pd.read_csv(f'{DATA_PATH}raw_data.csv')
    ed_df = pd.read_csv(f'{DATA_PATH}ep_df.csv')
    
    with open(f'{DATA_PATH}stopword.txt') as f:
        stopwords = literal_eval(f.read())
    
    with open(f'{DATA_PATH}topics.txt') as f:
        topics = literal_eval(f.read())
    
    return doc_df, ed_df, stopwords, topics

def dp_pipeline(doc_df, stopwords):
    ''' This function contains the pipeline for generating the 
    document-phrase matrix '''
    # Construct corpus (of tokens and noun phrases)
    corpus = doc_df['text'].values
    X_train = extractor.tokenise_doc(corpus, stopwords, max_phrase_len=3)
    
    # Generate TF for terms and noun phrases
    tf_terms = generator.generate_tf(X_train['tokens'])
    tf_phrases = generator.generate_tf(X_train['np'])
    
    # Generate document-phrase matrix
    dp_matrix = generator.generate_dp_matrix(tf_terms, tf_phrases, 
                                             doc_df['doc_id'], method="indirect")
    
    return pd.DataFrame(dp_matrix['matrix'].todense(),
                        index=dp_matrix['index'], columns=dp_matrix['columns'])

def dtopic_pipeline(dp_matrix, topics):
    ''' This function contains the pipeline for generating the 
    document-topic matrix'''
    # Load Scibert model
    MODEL_DIR = f'{MODEL_PATH}scibert_scivocab_uncased'
    model = BertModel.from_pretrained(MODEL_DIR)
    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    
    # Prepare model dictionary
    # Note: For the pretrained vectors of phrases, you will need to read here.
    # This example does not contain pretrained vectors
    model_dict = {
        'model': model,
        'tokenizer': tokenizer,
        'trained_vectors': None
    }
    
    # Generate document-topic matrix
    dtopic_matrix, topic_phrase = generator.generate_dtop_matrix(dp_matrix, topics, 
                                                                 model_dict, top_n=1)
    topic_vec = generator.generate_topic_vector(dtopic_matrix)
    dtopic_matrix = pd.DataFrame(dtopic_matrix['matrix'].todense(),
                                 index=dtopic_matrix['index'], 
                                 columns=dtopic_matrix['columns'])
    
    return dtopic_matrix, topic_vec, topic_phrase

def personalised_pipeline(ed_df, ed_matrix, dtopic_matrix, topic_vec):
    # Generate expert-document graph
    G = generator.generate_ecg(ed_df)
    
    # Generate personalised matrices 
    etop_matrix, dtop_matrix = generator.generate_pr_matrix(ed_matrix, 
                                                            dtopic_matrix, 
                                                            topic_vec['weights'].values, 
                                                            G, alpha=0.0)
    
    # Construct DataFrame
    etop_matrix = pd.DataFrame(etop_matrix['matrix'].todense(),
                               index=etop_matrix['index'],
                               columns=etop_matrix['columns'])
    dtop_matrix = pd.DataFrame(dtop_matrix['matrix'].todense(),
                               index=dtop_matrix['index'],
                               columns=dtop_matrix['columns'])
    
    return etop_matrix, dtop_matrix, G

def cv_pipeline(ed_matrix, ed_graph):
        # Generate co-occurrence expert-document
    exp_vec, doc_vec = generator.generate_ed_vector(ed_matrix, ed_graph)
    
    return exp_vec, doc_vec

def ef_pipeline(ed_matrix, ed_graph, exp_pr_df, doc_pr_df, ed_count, de_count):
    # Intialise parameters
    params = {
        'ed_graph': ed_graph,
        'ed_matrix': ed_matrix,
        'et_matrix': exp_pr_df,
        'dt_matrix': doc_pr_df,
        'lamb_e': 1.0,
        'lamb_d': 0.7,
        'max_iter': 5,
        'ed_count': ed_count,
        'de_count': de_count
    }
    topics = doc_pr_df.columns
    
    # Run model
    etop_matrix = trainer.run_expfinder(topics, params)
    
    return etop_matrix

if __name__ == "__main__":
    doc_df, ed_df, stopwords, topics = prepare_data()
    dp_matrix = dp_pipeline(doc_df, stopwords)
    dtopic_matrix, topic_vec, topic_phrase = dtopic_pipeline(dp_matrix, topics)
    exp_pr_df, doc_pr_df, ed_graph = personalised_pipeline(ed_df, dp_matrix, dtopic_matrix, topic_vec)
    ed_count, de_count = cv_pipeline(dp_matrix, ed_graph)
    etop_matrix = ef_pipeline(dp_matrix, ed_graph, exp_pr_df, doc_pr_df, ed_count, de_count)
    print(etop_matrix)
