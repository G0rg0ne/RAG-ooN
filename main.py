from loguru import logger  # Import loguru
import torch
import faiss
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import pdfplumber
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.cuda.empty_cache()
# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")  # Log the device being used


# Updated PDF text extraction function
# Step 1: Extract text from PDF files
def extract_text_from_pdfs(pdf_paths):
    texts = []
    for pdf_path in pdf_paths:
        logger.info(f"Extracting text from {pdf_path}...")  # Log progress
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
            texts.append(text)
        logger.info(f"Completed extraction from {pdf_path}.")
    return texts


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
    model_name = "gpt2-medium"  # The 125M version is optimized for CPU
    logger.info(f"Loading GPT-2 model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))

    # Step 1: Extract text from the PDFs
    texts = extract_text_from_pdfs(pdf_paths)

    # Step 2: Encode the texts into embeddings
    embeddings = encode_texts(texts, tokenizer, model, device)

    # Step 3: Build the FAISS index for fast retrieval
    index = build_faiss_index(embeddings)

    # Step 4: Retrieve the most relevant document based on the question
    context = search(question, index, tokenizer, model, texts)

    # Step 5: Use GPT-2 to generate an answer based on the context
    answer = generate_answer(question, context, tokenizer, model)

    logger.info("RAG system completed.")  # Log end of RAG system
    return answer


if __name__ == "__main__":
    # Example usage
    pdf_paths = [
        "data/world_cup_2022_final.pdf",
        "data/how_to_make_a_cake_recipe.pdf",
        "data/champions_league_2023_final.pdf",
    ]
    question = "What is the result of the 2022 world cup final"
    logger.info(f"Question: {question}")  # Log the question being asked
    answer = rag_system(pdf_paths, question)
    logger.info(f"Answer: {answer}")  # Log the answer generated
