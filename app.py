import streamlit as st
from loguru import logger  # Import loguru
from PyPDF2 import PdfReader
from htmlTemplates import css, bot_template, user_template
import os
import warnings
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util


def generate_answer(query, retrieved_chunks, context_chunks, max_new_tokens=200):
    """
    Generate an answer using the preloaded language model and tokenizer.
    """
    try:
        model = st.session_state.model
        tokenizer = st.session_state.tokenizer
        
        # Combine retrieved chunks into a context
        context = " ".join([
            f"[{chunk[0]}: Chunk {str(chunk[1] + 1)}] {context_chunks[chunk[0]][chunk[1]]}" 
            for chunk in retrieved_chunks
        ])
        
        # Prepare input text
        input_text = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(model.device)
        
        # Generate response
        logger.info("Generating response...")
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            eos_token_id=tokenizer.eos_token_id
        )
        
        # Decode and clean up the generated output
        answer = tokenizer.decode(output[0], skip_special_tokens=True)
        answer = answer.split("Answer:")[-1].strip() if "Answer:" in answer else answer.strip()
        return answer

    except Exception as e:
        logger.error(f"Error during answer generation: {e}")
        return "I'm sorry, I couldn't generate an answer. Please try again."
def retrieve(self, question, top_k=3):
    """
    Retrieve the most relevant chunks for the question.

    Args:
        question (str): The user's question.
        top_k (int): Number of top chunks to retrieve.

    Returns:
        list: List of tuples (filename, chunk_index, score).
    """
    # Retrieve fresh embeddings for the question
    question_embedding = self.retriever_model.encode(question, convert_to_tensor=True)
    
    # Retrieve top chunks based on similarity
    results = []
    for filename, embeddings in self.embeddings_by_file.items():
        scores = embeddings @ question_embedding.T
        top_results = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        results.extend([(filename, idx, score.item()) for idx, score in top_results])

    # Sort all results across files by score and return
    return sorted(results, key=lambda x: x[2], reverse=True)[:top_k]
def build_retriever(text_chunks):
    """
    Build the retriever model and create embeddings with document-level metadata.

    Args:
        text_chunks (dict): Dictionary with filenames as keys and lists of chunks as values.

    Returns:
        tuple: (embeddings_by_file, retriever_model)
    """
    retriever_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    embeddings_by_file = {}
    for filename, chunks in text_chunks.items():
        chunk_embeddings = retriever_model.encode(
            [f"{filename} | {chunk}" for chunk in chunks],  # Add file name to chunk text
            convert_to_tensor=True
        )
        embeddings_by_file[filename] = chunk_embeddings
    return embeddings_by_file, retriever_model
class CustomConversationChain:
    def __init__(self, retriever_model, embeddings_by_file):
        """
        Initialize the conversation chain.

        Args:
            retriever_model (SentenceTransformer): The model used to embed queries and chunks.
            embeddings_by_file (dict): Dictionary with filenames as keys and embedded chunks as values.
        """
        self.retriever_model = retriever_model
        self.embeddings_by_file = embeddings_by_file
        self.chat_history = []  # Stores the conversation history

    def add_to_history(self, role, message):
        """
        Add a message to the conversation history.

        Args:
            role (str): The role of the message sender ('user' or 'bot').
            message (str): The message content.
        """
        self.chat_history.append({"role": role, "message": message})
    
    def retrieve_chunks(self, question, top_k=3):
        """
        Retrieve the most relevant chunks for the question.

        Args:
            question (str): The user's question.
            top_k (int): Number of top chunks to retrieve.

        Returns:
            list: List of tuples (filename, chunk_index, score).
        """
        # Retrieve fresh embeddings for the question
        question_embedding = self.retriever_model.encode(question, convert_to_tensor=True)
        
        # Retrieve top chunks based on similarity
        results = []
        for filename, embeddings in self.embeddings_by_file.items():
            scores = embeddings @ question_embedding.T
            top_results = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
            results.extend([(filename, idx, score.item()) for idx, score in top_results])

        # Sort all results across files by score and return
        return sorted(results, key=lambda x: x[2], reverse=True)[:top_k]

    def format_context(self, retrieved_results, text_chunks):
        """
        Format the retrieved chunks into a context string.

        Args:
            retrieved_results (list): List of tuples (filename, chunk_index, score).
            text_chunks (dict): Dictionary where keys are filenames and values are lists of text chunks.

        Returns:
            str: Combined context string.
        """
        context = " ".join([
            f"[{filename}: Chunk {str(chunk_index + 1)}] {text_chunks[filename][chunk_index]}"
            for filename, chunk_index, score in retrieved_results
        ])
        return context

    def generate_response(self, question, text_chunks, top_k=3, max_new_tokens=200):
        """
        Generate a response to the user's question using the retrieved chunks.

        Args:
            question (str): The user's question.
            text_chunks (dict): Dictionary where keys are filenames and values are lists of text chunks.
            top_k (int): Number of top chunks to retrieve for context.
            max_new_tokens (int): Maximum number of tokens for the generated response.

        Returns:
            str: The generated answer.
        """
        # Retrieve relevant chunks
        retrieved_results = self.retrieve_chunks(question, top_k)

        # Format context from retrieved chunks
        context = self.format_context(retrieved_results, text_chunks)

        # Generate an answer using the context
        answer = generate_answer(question, retrieved_results, text_chunks, max_new_tokens)

        # Update conversation history
        self.add_to_history("user", question)
        self.add_to_history("bot", answer)

        return answer

    def get_chat_history(self):
        """
        Get the formatted conversation history.

        Returns:
            list: List of conversation messages.
        """
        return self.chat_history
def get_pdf_chunks(docs, chunk_size=500, overlap=50):
    """
    Extract text from PDF files and split it into chunks.

    Args:
        docs (list): List of PDF file paths.
        chunk_size (int): Maximum size of each text chunk.
        overlap (int): Number of overlapping characters between consecutive chunks.

    Returns:
        dict: A dictionary where keys are file names and values are lists of text chunks.
    """
    result = {}
    # Extract text from all pages of the PDF documents
    for pdf in docs:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Generate chunks with the specified size and overlap
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap

        # Store chunks in the dictionary with the file name as the key
        result[pdf.name] = chunks

    return result

def handle_question(question):
    # Ensure the conversation chain and chat history are initialized
    if "conversation_chain" not in st.session_state or st.session_state.conversation_chain is None:
        st.warning("Please process the PDF first to initialize the conversation chain.")
        return

    conversation_chain = st.session_state.conversation_chain

    # Generate response using the conversation chain
    text_chunks = st.session_state.text_chunks
    answer = conversation_chain.generate_response(question, text_chunks)

    # Display the chat history
    chat_history = conversation_chain.get_chat_history()
    for msg in chat_history:
        if msg["role"] == "user":
            st.write(user_template.replace("{{MSG}}", msg["message"]), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", msg["message"]), unsafe_allow_html=True)


def initialize_device():
    """
    Initialize the computation device (CUDA or CPU) only once and store in session state.
    """
    if "device" not in st.session_state:
        # Determine device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        st.session_state.device = device
        logger.info(f"Using device: {device}")  # Log only once during initialization

def load_model_and_tokenizer():
    """
    Load the language model and tokenizer only once and store them in session state.
    """
    if "model" not in st.session_state or "tokenizer" not in st.session_state:
        model_name = "microsoft/phi-2"
        logger.info(f"Loading model: {model_name}")
        
        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        st.session_state.model = model
        st.session_state.tokenizer = tokenizer
        logger.info("Model and tokenizer loaded and stored in session state.")

def main():
    st.set_page_config(page_title="RAG-ooN", page_icon=":brain:")
    st.header("RAG-ooN: PDF Intelligence System based on RAG :brain:")
    st.write(css, unsafe_allow_html=True)
    
    # Initialize device at startup
    initialize_device()

    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    load_model_and_tokenizer()  # Load model and tokenizer at startup
    question = st.text_input("Ask question from your document:")
    if question:
        handle_question(question)

    with st.sidebar:
        st.subheader("Your documents")
        docs = st.file_uploader("Upload your PDF here and click on 'Process'", accept_multiple_files=True)
        for uploaded_file in docs:
            st.write("Filename 📥:", uploaded_file.name)
        if st.button("Process 🔍 "):
            st.write("✅ PDF loaded")
            with st.spinner("Processing"):
                text_chunks = get_pdf_chunks(docs)
                embeddings_by_file, retriever_model = build_retriever(text_chunks)
                st.session_state.text_chunks = text_chunks
                st.session_state.conversation_chain = CustomConversationChain(retriever_model, embeddings_by_file)
                st.write("Processing complete. Ask your question in the input box.")
                
if __name__ == '__main__':
    warnings.filterwarnings('ignore')  # Suppress all other warnings
    os.environ['TRANSFORMERS_VERBOSITY'] = 'error'  # Suppress transformer warnings
    torch.cuda.empty_cache()
    # Check if GPU is available
    device = torch.device('cuda')
    logger.info(f"Using device: {device}")  # Log the device being used
    main()

