import os
import time
import logging
import pandas as pd
import sqlite3
from Bio import Entrez
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from sentence_transformers import SentenceTransformer, util
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import nltk
import torch
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

logging.basicConfig(level=logging.INFO)

# Data Ingestion
def preprocess_data(file_paths):
    dataframes = []
    for file in file_paths:
        df = pd.read_csv(file, delimiter='\t', encoding='ISO-8859-1', on_bad_lines='skip')
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        non_numeric_cols = df.select_dtypes(exclude=['float64', 'int64']).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        df[non_numeric_cols] = df[non_numeric_cols].fillna('')
        dataframes.append(df)
    return dataframes

def combine_text_data(dataframes, excel_df, pubmed_df):
    """Combine text data from multiple sources into a single list."""
    all_text_data = [text for df in dataframes for col in df.select_dtypes(include=[object]) for text in df[col].dropna().tolist()]
    
    # Extract text data from the Excel DataFrame
    for col in excel_df.select_dtypes(include=[object]).columns:
        all_text_data += excel_df[col].dropna().tolist()
    
    # Extract text data from the PubMed abstracts
    all_text_data += pubmed_df['Abstract'].tolist()
    
    return all_text_data

# Load and preprocess data
data_folder = os.getcwd()
file_paths = [os.path.join(data_folder, file) for file in os.listdir(data_folder) if file.endswith('.txt')]
dataframes = preprocess_data(file_paths)

for i, df in enumerate(dataframes):
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    non_numeric_cols = df.select_dtypes(exclude=['float64', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    df[non_numeric_cols] = df[non_numeric_cols].fillna('')
    dataframes[i] = df

# SQL query with corrected join conditions
query = '''
SELECT p.*, a.*, ad.*, t.*, msl.*
FROM table_6 p
LEFT JOIN table_2 a ON p.ApplNo = a.ApplNo
LEFT JOIN table_1 ad ON a.ApplNo = ad.ApplNo
LEFT JOIN table_10 t ON p.ApplNo = t.ApplNo
LEFT JOIN table_4 msl ON p.ApplNo = msl.ApplNo
LEFT JOIN table_5 ms ON msl.MarketingStatusID = ms.MarketingStatusID
'''

# Database connection and query execution
try:
    with sqlite3.connect('database.db') as conn:
        for i, df in enumerate(dataframes):
            table_name = f"table_{i}"
            df.to_sql(table_name, conn, index=False, if_exists='replace')

        final_df = pd.read_sql_query(query, conn)

        # Load the Excel file from the same folder as the Jupyter Notebook
        excel_file_path = os.path.join(data_folder, 'EMA - medicines_output_medicines_en.xlsx')
        excel_df = pd.read_excel(excel_file_path)

except (sqlite3.Error, Exception) as e:
    logging.error(f"Database error: {e}")
    final_df = pd.DataFrame()  # Empty DataFrame in case of error
    excel_df = pd.DataFrame()  # Empty DataFrame in case of error

# PubMed Data Extraction
Entrez.email = "giuseppe.farrugia.14@um.edu.mt"
search_term = "approved drug"
retmax = 10000
current_year = 2024
start_year = current_year - 5

# Fetch all article IDs
handle = Entrez.esearch(db="pubmed", term=search_term, retmax=retmax, mindate=f"{start_year}-01-01", maxdate=f"{current_year}-12-31")
record = Entrez.read(handle)
id_list = record["IdList"]

def fetch_article_metadata(article_id):
    try:
        handle = Entrez.efetch(db="pubmed", id=article_id, rettype="abstract", retmode="xml")
        article_record = Entrez.read(handle)
        return article_record
    except HTTPError as e:
        print(f"HTTPError for article ID {article_id}: {e}")
        return None

batch_size = 10
articles_metadata = []
for i in tqdm(range(0, len(id_list), batch_size), desc="Fetching article metadata"):
    batch_ids = id_list[i:i+batch_size]
    for article_id in batch_ids:
        metadata = fetch_article_metadata(article_id)
        if metadata:
            articles_metadata.append(metadata)
        time.sleep(0.1)

def extract_info(article_metadata):
    article_info = {}
    if 'PubmedArticle' in article_metadata and article_metadata['PubmedArticle']:
        article_info['Title'] = article_metadata['PubmedArticle'][0]['MedlineCitation']['Article']['ArticleTitle']
        if 'Abstract' in article_metadata['PubmedArticle'][0]['MedlineCitation']['Article']:
            article_info['Abstract'] = article_metadata['PubmedArticle'][0]['MedlineCitation']['Article']['Abstract']['AbstractText'][0]
        else:
            article_info['Abstract'] = "No abstract available"
    else:
        article_info['Title'] = "No title available"
        article_info['Abstract'] = "No abstract available"
    return article_info

articles_info = []
for metadata in tqdm(articles_metadata, desc="Extracting article info"):
    articles_info.append(extract_info(metadata))

pubmed_df = pd.DataFrame(articles_info)

# Combine all textual data into a single list
all_text_data = combine_text_data(dataframes, excel_df, pubmed_df)

# Generate Embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
batch_size = 500
pubmed_embeddings = []

for i in tqdm(range(0, len(all_text_data), batch_size)):
    batch = all_text_data[i:i + batch_size]
    embeddings = model.encode(batch, convert_to_numpy=True)
    pubmed_embeddings.append(embeddings)

pubmed_embeddings = np.vstack(pubmed_embeddings)
embedding_dim = pubmed_embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)

def add_to_faiss_index(index, data, batch_size):
    for i in tqdm(range(0, len(data), batch_size), desc="Adding to FAISS index"):
        index.add(data[i:i + batch_size])

add_to_faiss_index(index, pubmed_embeddings, batch_size)
print(f"FAISS index size: {index.ntotal}")

# Sparse retrieval using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(all_text_data)

def retrieve_dense(query, index, model, top_k=5):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    return distances[0], indices[0]

def retrieve_sparse(query, tfidf_matrix, vectorizer, top_k=5):
    query_vec = vectorizer.transform([query])
    scores = np.dot(tfidf_matrix, query_vec.T).toarray()
    top_k_indices = np.argsort(scores, axis=0)[-top_k:][::-1].flatten()
    return top_k_indices

def combine_retrievals(dense_indices, sparse_indices, all_text_data, max_contexts=5):
    dense_indices = dense_indices[1]

    try:
        dense_indices = set(map(int, dense_indices))
        sparse_indices = set(map(int, sparse_indices))
    except TypeError as e:
        print("TypeError during conversion:", e)
        return []

    unique_indices = list(dense_indices | sparse_indices)
    
    valid_contexts = []
    seen_contexts = set()
    for idx in unique_indices:
        if idx < len(all_text_data):
            context = all_text_data[idx]
            if context not in seen_contexts:
                seen_contexts.add(context)
                valid_contexts.append(context)

    return valid_contexts[:max_contexts]

# Generate Response
def generate_response(query, dense_indices, sparse_indices, all_text_data, max_context_length=512):
    """Generate a response based on the given query and combined retrieval results."""
    valid_contexts = combine_retrievals(dense_indices, sparse_indices, all_text_data, max_contexts=5)

    if not valid_contexts:
        valid_contexts = ["Model couldn't generate results."]

    combined_context = "\n\n".join(valid_contexts)
    query_with_context = f"Based on the information below, provide a concise and informative answer in about 200 words.\n\n{combined_context}\n\nQuestion: {query}\nAnswer:"

    inputs = tokenizer.encode_plus(
        query_with_context,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=max_context_length
    )
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    output_ids = gen_model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=input_ids.shape[1] + 100,
        pad_token_id=tokenizer.eos_token_id,  # Set explicit pad token ID
        num_beams=5,  # Beam search with 5 beams
        no_repeat_ngram_size=2
    )

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    # Remove unnecessary characters
    generated_text = re.sub(r'[^a-zA-Z0-9\s,.!?\'"-]', '', generated_text)

    # Ensure proper sentence formation and truncate to around 200 words
    sentences = generated_text.split('. ')
    truncated_response = '. '.join(sentences[:6])  # Adjust number of sentences as needed
    if len(truncated_response.split()) > 200:
        truncated_response = ' '.join(truncated_response.split()[:200]) + '...'
    if truncated_response and truncated_response[-1] != '.':
        truncated_response += '.'

    # Format the response
    formatted_response = f"**Question:** {query}\n\n"
    formatted_response += f"**Answer:**\n\n"
    formatted_response += f"{truncated_response}"

    return formatted_response


# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gen_model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set the padding token

def query_rag(user_query):
    """Retrieve dense and sparse results and generate a response for the given query."""
    dense_results = retrieve_dense(user_query, index, model, top_k=10)
    sparse_results = retrieve_sparse(user_query, tfidf_matrix, vectorizer, top_k=10)
    response = generate_response(user_query, dense_results, sparse_results, all_text_data)
    return response
