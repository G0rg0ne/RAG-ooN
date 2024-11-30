from PyPDF2 import PdfReader
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import yaml

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import faiss
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import pdfplumber

# Updated PDF text extraction function
# Step 1: Extract text from PDF files
def extract_text_from_pdfs(pdf_paths):
    texts = []
    for pdf_path in pdf_paths:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
            texts.append(text)
    return texts
def encode_texts(texts, tokenizer, model):
    input_ids = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)["input_ids"]
    with torch.no_grad():
        outputs = model.transformer(input_ids)
    # Use the embeddings from the last hidden state
    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embeddings

# Step 3: Build a retrieval system using FAISS
def build_faiss_index(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance metric
    index.add(embeddings)
    return index

# Step 4: Search for the most relevant document based on a query
def search(query, index, tokenizer, model, texts):
    query_embedding = encode_texts([query], tokenizer, model)
    D, I = index.search(query_embedding, k=1)  # Retrieve the closest document
    return texts[I[0][0]]  # Return the most relevant document

# Step 5: Use GPT-2 to generate an answer based on the retrieved document
def generate_answer(question, context, tokenizer, model):
    input_text = f"Question: {question}\nContext: {context}\nAnswer:"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    
    # Generate a response using GPT-2
    output_ids = model.generate(input_ids, max_length=250, num_return_sequences=1, no_repeat_ngram_size=2,do_sample=True)
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return answer

# Main function
def rag_system(pdf_paths, question):
    # Load the pre-trained GPT-2 model and tokenizer
    model_name = "EleutherAI/gpt-neo-2.7B"  # The 125M version is optimized for CPU
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    import pdb;pdb.set_trace()
    # Step 1: Extract text from the PDFs
    texts = extract_text_from_pdfs(pdf_paths)

    # Step 2: Encode the texts into embeddings
    embeddings = encode_texts(texts, tokenizer, model)

    # Step 3: Build the FAISS index for fast retrieval
    index = build_faiss_index(embeddings)

    # Step 4: Retrieve the most relevant document based on the question
    context = search(question, index, tokenizer, model, texts)

    # Step 5: Use GPT-2 to generate an answer based on the context
    answer = generate_answer(question, context, tokenizer, model)
    return answer

if __name__ == "__main__":
    # Example usage
    pdf_paths = ["data/world_cup_2022_final.pdf", "data/how_to_make_a_cake_recipe.pdf", "data/champions_league_2023_final.pdf"]
    question = "what is the world cup match date?"
    answer = rag_system(pdf_paths, question)
    print(answer)

