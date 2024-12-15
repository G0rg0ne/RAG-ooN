from loguru import logger  # Import loguru
import torch
import faiss
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import pdfplumber
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import warnings
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util
import re


def split_pdf_into_chunks(pdf_path, chunk_size=500, overlap=50):
    """
    Splits a PDF file into text chunks suitable for a RAG system.

    Args:
        pdf_path (str): Path to the PDF file.
        chunk_size (int): Maximum number of words per chunk.
        overlap (int): Number of overlapping words between consecutive chunks.

    Returns:
        list: A list of text chunks.
    """
    # Read the PDF
    reader = PdfReader(pdf_path)
    
    # Extract text from all pages
    text = "\n".join(page.extract_text() for page in reader.pages)

    # Clean the text
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace

    # Split text into words
    words = text.split()

    # Create chunks
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))

    return chunks

def process_list_of_pdfs(pdf_paths, chunk_size=500, overlap=50):
    """
    Process a list of PDF files and split them into text chunks.

    Args:
        pdf_paths (list): List of paths to the PDF files.
        chunk_size (int): Maximum number of words per chunk.
        overlap (int): Number of overlapping words between consecutive chunks.

    Returns:
        dict: A dictionary where keys are PDF filenames and values are lists of text chunks.
    """
    chunks_by_file = {}

    for pdf_path in pdf_paths:
        filename = pdf_path.split("/")[-1]  # Extract filename from path
        chunks = split_pdf_into_chunks(pdf_path, chunk_size, overlap)
        chunks_by_file[filename] = chunks

    return chunks_by_file


def encode_texts(texts, tokenizer, model, device):
    
    logger.debug("Encoding texts into embeddings...")  # Debug log

    # Tokenize the texts and move the input tensor to the specified device (GPU/CPU)
    input_ids = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=512
    )["input_ids"]
    input_ids = input_ids.to(device)  # Move input tensor to the device (GPU/CPU)

    # Ensure the model is on the correct device (GPU or CPU)
    model = model.to(device)  # Move model to device

    # Perform forward pass on the model with output_hidden_states=True to get hidden states
    with torch.no_grad():  # Disable gradient computation to save memory during inference
        outputs = model(input_ids, output_hidden_states=True)

    # Get the hidden states from the last layer
    hidden_states = outputs.hidden_states[-1]  # The last layer's hidden states

    # Use the embeddings from the last hidden state (average across token embeddings)
    embeddings = (
        hidden_states.mean(dim=1).cpu().numpy()
    )  # Move embeddings to CPU if needed

    logger.debug(f"Encoded {len(texts)} texts into embeddings.")
    return embeddings


# Step 3: Build a retrieval system using FAISS
def build_faiss_index(embeddings):
    logger.info("Building FAISS index...")  # Log index creation
    index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance metric
    index.add(embeddings)
    logger.info("FAISS index built successfully.")
    return index


# Step 4: Search for the most relevant document based on a query
def search(query, index, tokenizer, model, texts):
    logger.info(
        f"Searching for the most relevant document for query: {query}"
    )  # Log search process
    query_embedding = encode_texts([query], tokenizer, model, device)
    D, I = index.search(query_embedding, k=1)  # Retrieve the closest document
    logger.info("Search completed. Retrieved the most relevant document.")
    return texts[I[0][0]]  # Return the most relevant document


# Step 5: Use GPT-2 to generate an answer based on the retrieved document
def generate_answer(question, context, tokenizer, model):
    logger.info("Generating answer  ...")  # Log answer generation
    input_text = f"Question: {question}\nContext: {context}\nAnswer:"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(
        device
    )  # Move to GPU

    # Generate a response using GPT-2
    # output_ids = model.generate(input_ids, max_length=350, num_return_sequences=1, no_repeat_ngram_size=2, do_sample=False)
    output_ids = model.generate(
        input_ids,
        max_length=300,  # Max length of the generated answer
        num_beams=5,  # Use beam search with 5 beams
        no_repeat_ngram_size=2,  # Prevent repeating n-grams for diversity
        early_stopping=True,  # Stop generating if a complete answer is found
        do_sample=False,  # Use beam search, not sampling
        temperature=1,  # Control the randomness, 1.0 is default
    )
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    logger.info("Answer generated.")
    return answer


# Main function
def rag_system(pdf_paths, question):
    
    logger.info("Starting the RAG system...")  # Log start of RAG system
    # Load the pre-trained GPT-2 model and tokenizer
    model_name = "microsoft/phi-2"  # The 125M v ersion is optimized for CPU
    logger.info(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    #if tokenizer.pad_token is None:
    #    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    #    model.resize_token_embeddings(len(tokenizer))
    
    # Step 2: Encode the texts into embeddings
    embeddings = encode_texts(pdf_paths, tokenizer, model, device)

    # Step 3: Build the FAISS index for fast retrieval
    index = build_faiss_index(embeddings)

    # Step 4: Retrieve the most relevant document based on the question
    context = search(question, index, tokenizer, model, pdf_paths)

    # Step 5: Use GPT-2 to generate an answer based on the context
    answer = generate_answer(question, context, tokenizer, model)

    logger.info("RAG system completed.")  # Log end of RAG system
    return answer

def build_retriever(text_chunks):
    """
    Build embeddings for the text chunks using a retriever model.

    Args:
        text_chunks (dict): Dictionary where keys are filenames and values are lists of text chunks.

    Returns:
        dict, SentenceTransformer: A dictionary of embeddings by file and the retriever model.
    """
    retriever_model = SentenceTransformer('all-MiniLM-L6-v2')  # Example model
    embeddings_by_file = {}

    for filename, chunks in text_chunks.items():
        embeddings = retriever_model.encode(chunks, convert_to_tensor=True)
        embeddings_by_file[filename] = embeddings

    return embeddings_by_file, retriever_model

def retrieve(query, embeddings_by_file, retriever_model, top_k=3):
    """
    Retrieve the most relevant chunks for a given query.

    Args:
        query (str): The input query string.
        embeddings_by_file (dict): Dictionary with filenames as keys and embedded chunks as values.
        retriever_model (SentenceTransformer): The model used to embed the query.
        top_k (int): Number of top matches to return.

    Returns:
        list: A list of tuples containing the filename, chunk, and score of the top matches.
    """
    query_embedding = retriever_model.encode(query, convert_to_tensor=True)
    results = []

    for filename, embeddings in embeddings_by_file.items():
        scores = util.cos_sim(query_embedding, embeddings)[0]
        top_results = torch.topk(scores, k=min(top_k, len(scores)))

        for score, idx in zip(top_results.values, top_results.indices):
            results.append((filename, idx.item(), score.item()))

    # Sort results by score
    results = sorted(results, key=lambda x: x[2], reverse=True)
    return results[:top_k]

def generate_answer(query, retrieved_chunks, context_chunks, max_new_tokens=200):
    """
    Generate an answer using the language model and retrieved chunks.

    Args:
        query (str): The input query.
        retrieved_chunks (list): List of tuples (filename, chunk_index, score) from the retriever.
        context_chunks (dict): Dictionary of all chunks by filename.
        model: The causal language model.
        tokenizer: The tokenizer for the model.
        max_length (int): Maximum length of the generated response.

    Returns:
        str: The generated answer.
    """
    
    model_name = "microsoft/phi-2"  # The 125M v ersion is optimized for CPU
    logger.info(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # Combine the retrieved chunks into a context
    context = " ".join([f"[{chunk[0]}: Chunk {chunk[1] + 1}] {context_chunks[chunk[0]][chunk[1]]}" for chunk in retrieved_chunks])

    # Prepare input for the model
    input_text = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(model.device)
    
    # Generate a response
    output = model.generate(**inputs, max_new_tokens=max_new_tokens, eos_token_id=tokenizer.eos_token_id)
    # Decode the output
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer

if __name__ == "__main__":
    warnings.filterwarnings('ignore')  # Suppress all other warnings
    os.environ['TRANSFORMERS_VERBOSITY'] = 'error'  # Suppress transformer warnings

    torch.cuda.empty_cache()
    # Check if GPU is available
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device('cuda')
    logger.info(f"Using device: {device}")  # Log the device being used

    # Example usage
    pdf_paths = [
        "data/world_cup_2022_final.pdf",
        "data/how_to_make_a_cake_recipe.pdf",
        "data/champions_league_2023_final.pdf",
    ]
    all_chunks = process_list_of_pdfs(pdf_paths, chunk_size=500, overlap=50)
    embeddings_by_file, retriever_model = build_retriever(all_chunks)
    # Query the retriever
    query = "Who scored in the Champions league 2023 final ? "
    retrieved_chunks = retrieve(query, embeddings_by_file, retriever_model, top_k=1)
    logger.info(f"Question: {query}")  # Log the question being asked
    answer = generate_answer(query, retrieved_chunks, all_chunks)
    logger.info(f"Answer: {answer}")  # Log the answer generated
