import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def create_icd_csv():
    # Absolute Path
    path_to_icd = "./ICD-10 Codes/icd10cm-codes-2025.txt"

    with open(path_to_icd, 'r') as icd:
        all_icd = icd.read().strip().split('\n')

    icd_split = [icd.split(' ', 1) for icd in all_icd] # Split into [ICD, Description] 
    icd_clean = [[icd.strip() for icd in icd_pair] for icd_pair in icd_split] # Strip trailing spaces in strings

    icd_csv = pd.DataFrame(icd_clean, columns = ["Code", "Description"])
    icd_csv.to_csv('./Saved_Data/icd10.csv', index = False)

def read_icd():
    return pd.read_csv('./Saved_Data/icd10.csv')

def mean_pooling(inputs, model):
    with torch.no_grad():
            outputs = model(**inputs)
            last_hidden_state = outputs.last_hidden_state
            
    attention_mask = inputs['attention_mask']
    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()

    sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
    
    mean_pooling = sum_embeddings / sum_mask  # shape: [batch_size, hidden_dim]

    return mean_pooling

def create_icd_embeddings(model_name, file_name, model_type):
    icd = read_icd()

    if model_type == 's':
        model = SentenceTransformer(model_name)
        embeddings = {row.Code: model.encode(row.Description) for row in icd.itertuples()}
    
        save_icd_embeddings(file_name, embeddings)

        return
    
    if model_type == 'w':
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        embeddings = {}

        for row in icd.itertuples():
            input = tokenizer(row.Description, padding = True, truncation = True, return_tensors = 'pt')
            embeddings[row.Code] = mean_pooling(input, model)
        
        save_icd_embeddings(file_name, embeddings)

        return

def embed_text(model_name, model_type, text):
    if model_type == 's':
        model = SentenceTransformer(model_name)

        return model.encode(text)

def save_icd_embeddings(file_name, embeddings):
    with open(f"./Saved_Data/ICD_Pickles/{file_name}.pkl", "wb") as p:
        pickle.dump(embeddings, p)

def open_icd_embeddings(file_name):
    with open(f"./Saved_Data/ICD_Pickles/{file_name}.pkl", "rb") as p:
        return pickle.load(p)

def get_top_k_similar(text_embedding, icd_embeddings, k = 5):
    codes = list(icd_embeddings.keys())
    embeddings = np.array([icd_embeddings[code] for code in codes])

    text_embedding = np.array(text_embedding).reshape(1, -1)

    sims = cosine_similarity(text_embedding, embeddings).flatten()

    top_k_codes_index = sims.argsort()[::-1][:k]

    top_k_codes = [codes[index] for index in top_k_codes_index]
    top_k_sims = [sims[index] for index in top_k_codes_index] 

    return top_k_codes, top_k_sims
