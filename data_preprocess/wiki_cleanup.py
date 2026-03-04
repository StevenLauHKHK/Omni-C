import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer
from transformers import BertTokenizer
import torchaudio
import random
import os
import re
import json
import tqdm
import numpy as np

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Clean text
# Clean WikiText content
def clean_wikitext(text):
    """
    Clean WikiText content by removing Wikipedia markup, citations, headers,
    special tokens, HTML links, and extra whitespace while preserving 
    proper sentence structure.
    """
    # Remove wiki internal links [[...]]
    text = re.sub(r'\[\[.*?\]\]', '', text)
    
    # Remove templates/infoboxes {{...}}
    text = re.sub(r'\{\{.*?\}\}', '', text)
    
    # Remove section headers (= Title =, == Subtitle ==, etc.)
    text = re.sub(r'==[^=]*==+', '', text)
    text = re.sub(r'=[^=]*=+', '', text)
    
    # Remove citations and references [1], [2], etc.
    text = re.sub(r'\[\d+\]', '', text)
    
    # Remove HTML tags and links
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove any URL or link remnants
    text = re.sub(r'www\.[a-zA-Z0-9./-]+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def decode_tokens(tokenizer, token_ids):
    """
    Convert token IDs back to text.
    
    Args:
        token_ids: NumPy array or list of token IDs
        
    Returns:
        Decoded text string
    """
    # Remove padding tokens (0) if any
    if isinstance(token_ids, np.ndarray):
        token_ids = token_ids.tolist()
        
    # Filter out padding tokens
    valid_tokens = [t for t in token_ids if t != 0]
    
    # Decode back to text
    decoded_text = tokenizer.decode(valid_tokens, skip_special_tokens=True)
    return decoded_text

def preprocess_wikitext(input_dir, output_path, max_len=256, split="train", chunk_size=512):
    """
    Preprocess WikiText data for masked autoencoder training.
    
    Args:
        input_dir: Directory containing WikiText files
        output_path: Output directory
        max_len: Maximum sequence length
        split: 'train', 'valid', or 'test'
        chunk_size: Number of tokens to chunk the text into before tokenizing
    """
    # Load BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    file_names = [f for f in os.listdir(input_dir)]
    for file_name in file_names:

        # Read the WikiText file
        file_path = os.path.join(input_dir, file_name)
        subdir = os.path.basename(os.path.dirname(file_path))

        print(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Clean the text
        text = clean_wikitext(text)
 
        # Split text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s for s in sentences if len(s.strip()) > 0]
    
        # Chunk sentences into documents
        documents = []
        current_doc = ""
    
        for sentence in sentences:
            # print(sentence, len(current_doc), len(sentence))
            if len(current_doc) + len(sentence) > chunk_size:
                # print(current_doc)
                if current_doc:
                    # print('in=', current_doc)
                    documents.append(current_doc)
                current_doc = sentence
            else:
                if current_doc:
                    current_doc += " " + sentence
                else:
                    current_doc = sentence
        
        if current_doc:  # Add the last document
            documents.append(current_doc)

        # Process documents
        datalist = []
        


        for idx, document in enumerate(documents):
            # Assign a dum-label (-1) based on document index
            label = -1
        
            # Tokenize the document
            encoded = tokenizer.encode_plus(
                document,
                add_special_tokens=True,
                max_length=max_len,
                padding='max_length',
                truncation=True,
                return_tensors='np'
            )
            token = np.array(encoded['input_ids'][0])
            # Save embeddings
            output_file_name = f"{file_name}_{idx}.npy"
            
            data_dict = {
                "name": f"{subdir}/{output_file_name}",
                "label": int(-1)
            }
            datalist.append(data_dict)
            
            # Save the tokenized document
            saving_subdir = os.path.join(output_path, subdir)
            os.makedirs(saving_subdir, exist_ok=True)
            saving_path = os.path.join(saving_subdir, output_file_name)
            np.save(saving_path, token)
    
    # Save metadata
    return datalist
    

# Save vocabulary for later use
def save_vocabulary(output_path):
    vocab_path = os.path.join(output_path, 'bert_vocab.json')
    if not os.path.exists(vocab_path):
        vocab_dict = dict(tokenizer.vocab)
        with open(vocab_path, 'w') as f:
            json.dump(vocab_dict, f, indent=2)
        print(f"Saved vocabulary with {len(vocab_dict)} tokens to {vocab_path}")

def create_json_file(wiki_dir, wiki_data_dir, wiki_output_npy_dir):
    # create json file for train
    output_json_path = os.path.join(wiki_dir, 'wikitext_train.json')
    subdirs = [d for d in os.listdir(wiki_output_npy_dir) if os.path.isdir(os.path.join(wiki_output_npy_dir, d))]
    
    datalists = []
    for subdir in subdirs:
        subdir_path = os.path.join(wiki_output_npy_dir, subdir)
        print(subdir_path)
        documents = os.listdir(subdir_path)
        for document in documents:
            data_dict = {
                "name": f"{subdir}/{document}",
                "label": -1  # Dummy label
            }
            datalists.append(data_dict)
    with open(output_json_path, "w") as f:
        json.dump(datalists, f, indent=2)
    print(output_json_path)


# Process documents
datalist = []
wiki_dir = '/data_ssd/DATA/Wikipedia'
wiki_data_dir = '/data_ssd/DATA/Wikipedia/enwiki_text'
wiki_output_npy_dir = '/data_ssd/DATA/Wikipedia/enwiki_npy'
max_length = 128  # Max sequence length for tokens

# Save BERT vocabulary
save_vocabulary(wiki_dir)

subdirs = [d for d in os.listdir(wiki_data_dir) if os.path.isdir(os.path.join(wiki_data_dir, d))]

datalists = []
for subdir in subdirs:
    subdir_path = os.path.join(wiki_data_dir, subdir)
    datalist = preprocess_wikitext(subdir_path, wiki_output_npy_dir)
    datalists.extend(datalist)

# Save the processed data to JSON
json_file_name = f"wikitext_train.json"
with open(os.path.join(wiki_data_dir, json_file_name), "w") as f:
    json.dump(datalists, f, indent=2)


create_json_file(wiki_dir, wiki_data_dir, wiki_output_npy_dir)




