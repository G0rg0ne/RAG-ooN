import streamlit as st
from RAG_system import build_retriever,retrieve,generate_answer
from loguru import logger  # Import loguru
from PyPDF2 import PdfReader
from htmlTemplates import css, bot_template, user_template
import os
import warnings
import torch

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

    def generate_response(self, question, text_chunks, top_k=3):
        """
        Generate a response to the user's question using the retrieved chunks.

        Args:
            question (str): The user's question.
            text_chunks (dict): Dictionary where keys are filenames and values are lists of text chunks.
            top_k (int): Number of top chunks to retrieve for context.

        Returns:
            str: The generated answer.
        """
        # Retrieve the most relevant chunks for the question
        retrieved_results = retrieve(question, self.embeddings_by_file, self.retriever_model, top_k=top_k)

        # Extract the actual chunks from the retrieval results
        relevant_chunks = []
        for filename, idx, score in retrieved_results:
            relevant_chunks.append(text_chunks[filename][idx])

        # Combine the context from the retrieved chunks
        context = " ".join(relevant_chunks)

        # Use the model to generate an answer (assuming generate_answer is defined)
        answer = generate_answer(question, relevant_chunks, text_chunks)

        # Update the conversation history
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
    response=st.session_state.conversation({'question': question})
    st.session_state.chat_history=response["chat_history"]
    for i,msg in enumerate(st.session_state.chat_history):
        if i%2==0:
            st.write(user_template.replace("{{MSG}}",msg.content,),unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}",msg.content),unsafe_allow_html=True)
def main():
    st.set_page_config(page_title="RAG-ooN",page_icon=":brain:")
    st.header("RAG-ooN: PDF Intelligence System based on RAG :brain:")
    st.write(css,unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation=None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history=None
    question=st.text_input("Ask question from your document:")
    if question and st.session_state.conversation_chain:
        # Use the custom conversation chain to generate a response
        response = st.session_state.conversation_chain.generate_response(question, st.session_state.text_chunks)
        # Display the conversation
        for msg in st.session_state.conversation_chain.get_chat_history():
            if msg["role"] == "user":
                st.write(user_template.replace("{{MSG}}", msg["message"]), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", msg["message"]), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your documents")
        docs=st.file_uploader("Upload your PDF here and click on 'Process'",accept_multiple_files=True)
        #pdf_paths = []
        for uploaded_file in docs:
            st.write("Filename 📥:", uploaded_file.name)
            #with open(uploaded_file.name, "wb") as f:  
            #    f.write(uploaded_file.read())
            #    pdf_paths.append(uploaded_file.name)
        if st.button("Process 🔍 "):
            st.write("✅ PDF loaded")
            with st.spinner("Processing"):
                text_chunks=get_pdf_chunks(docs)
                embeddings_by_file, retriever_model = build_retriever(text_chunks)
                if not isinstance(embeddings_by_file, dict):
                    raise ValueError("Embeddings by file must be a dictionary.")

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

# st.write("✅ PDF loaded")
# st.write("📄 PDF loaded")
# st.write("🎉 PDF loaded successfully!")
# st.write("🚀 PDF processing started")
# st.write("📥 File uploaded")
# st.write("⚠️ Error loading PDF")
# st.write("🔍 Analyzing PDF content")